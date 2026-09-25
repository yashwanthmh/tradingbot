"""A language model as one more source of candidate specs — and nothing more.

The adapter's whole contract is `SpecProposer`. It returns `Proposal`s, each
holding a spec `StrategySpec.parse` accepted, and it holds no ledger, no bars,
no broker and no gate. What the model may be told is decided in `prompts`,
which builds both messages from abstract types and audits the rendered text
before it leaves the process. What the model says back is untrusted data, and
four things follow from that.

**The reply is parsed as JSON and nothing else.** Every number becomes a
`Decimal`, so `1e999999999` is a value the schema refuses rather than an `inf`
that compares false against everything, and `NaN` or `Infinity` become Decimals
the schema refuses *per item* rather than a parse error that loses the whole
reply. Nesting past the parser's recursion limit is a refused reply, not a
crash. There is no `eval`, `exec`, `ast.literal_eval` or template engine
anywhere on the path, and the fuzzing suite's audit hook holds that for this
module as it does for the interpreter.

**The model decides three fields.** `entry`, `exit` and `expected_edge_bps` are
read from each item; `name`, `notes` and `min_holding_minutes` are set here.
The holding period is the one that matters — a model free to declare `0` could
propose exactly the high-turnover specs the fee schedule forbids — and the
declared edge still has to clear the validator's band like any other
proposer's. Keys the model adds are ignored and reported, never obeyed.

**A reply the search cannot use fails the call; it does not become a random
search.** A refusal, a transport failure, a reply with no JSON array, or one in
which no item validates raises before anything is evaluated — so nothing is
recorded as a trial and the operator sees why. Substituting random draws would
run a different search from the one asked for, under its name. Items refused
inside an otherwise usable reply are counted and reported: they never became
specs, so they have no hash to record and cannot be selected, and selection is
the thing multiplicity accounting exists to price.

**It is not reproducible from the seed, and says so.** The random and mutation
proposers replay exactly from `SearchBudget.seed`; a model does not. So the
exchange is recorded instead — each call leaves an `LLMReport`, which the cycle
ledgers as `search.specs_proposed` with the exact prompts, a hash of the
reply and every accepted spec. A candidate from this path can be reconstructed
from the ledger even though it cannot be regenerated.

The Anthropic SDK is an optional extra (`uv sync --extra llm`), imported only
when an `AnthropicClient` is built, so the rest of the research path neither
needs nor loads it.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from tb.core.canonical import hash_payload, sha256_hex
from tb.ops.secrets import LIVE_KEY_VAR
from tb.registry.models import AuthorKind
from tb.research.llm.prompts import (
    RegimeDescription,
    audit_prompt,
    render_system_prompt,
    render_user_prompt,
)
from tb.research.mutate import Proposal, ProposalBounds, ProposalError
from tb.strategy.dsl.schema import SpecError, StrategySpec

DEFAULT_MODEL = "claude-opus-5"
# The output budget for one call. Models that think by default spend part of it
# thinking, and a truncated reply loses its tail, so it is sized well above what
# `MAX_SPECS_PER_CALL` specs need rather than at it.
DEFAULT_MAX_TOKENS = 32_000
DEFAULT_TIMEOUT_SECONDS = 600.0
# The beta that gates the scalar `fallbacks: "default"` form. The array form is
# gated by a different header, and pairing either header with the other form is
# refused by the API — so the two are not interchangeable spellings.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Most specs one call asks for. One call per search is the cost model, and a
# request for more than fit comfortably in the output budget comes back cut
# off, which loses the items at the end — the ones a model tends to make most
# different from the rest.
MAX_SPECS_PER_CALL = 40
# The longest reply read at all. Far beyond what `DEFAULT_MAX_TOKENS` produces;
# a bound so nothing upstream can hand the parser an unbounded string.
MAX_REPLY_CHARS = 400_000
# Refusal reasons kept for the report, and how long each may be. The count of
# refusals is exact; the list is for reading.
MAX_REASONS_KEPT = 50
MAX_REASON_CHARS = 300

# The fields the model decides. Everything else on a spec is set here.
MODEL_KEYS: frozenset[str] = frozenset({"entry", "exit", "expected_edge_bps"})


class LLMError(ProposalError):
    """The model did not supply usable proposals this time."""


class LLMUnavailable(LLMError):
    """The model cannot be called from this process as it is configured."""


# --------------------------------------------------------------------------
# The call
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """One exchange, reduced to what the proposer reads.

    `served_model` is the model that wrote the text. It differs from
    `requested_model` when a refusal fallback ran, which is worth recording:
    the two carry different classifiers and write different ideas.
    """

    text: str
    requested_model: str
    served_model: str
    stop_reason: str | None = None
    fell_back: bool = False
    # The category and explanation the API gave, when `stop_reason` is
    # "refusal". Empty otherwise.
    refusal: str = ""


class LLMClient(Protocol):
    """The single call the proposer makes. A fake in tests, the Claude API for real."""

    @property
    def model(self) -> str: ...

    def complete(self, *, system: str, user: str) -> LLMResponse: ...


class AnthropicClient:
    """The Claude API through the official SDK, behind `LLMClient`.

    Streams and collects the final message rather than making a blocking
    request, because a reply with thinking enabled can run long enough to
    meet the SDK's guard against long non-streaming calls.

    **Refusal fallbacks are on by default** (`fallbacks="default"`). A
    classifier can decline a benign request; with the fallback, the API re-runs
    it on Anthropic's recommended substitute inside the same call instead of
    returning the refusal. The substitute's name lands in `served_model`, so a
    spec's provenance says which model actually wrote it. `fallbacks=False`
    turns it off for a model or platform that does not offer it.

    **It refuses to exist in a process holding `T212_LIVE_API_KEY`.** This is
    the one research component that sends text to a third party, and the rule
    is that research processes run without the live key at all. Enforcing it
    where the text leaves is cheaper than trusting every caller's environment.
    The API key itself is read by the SDK from `ANTHROPIC_API_KEY` and never
    passes through this module.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        fallbacks: bool = True,
    ) -> None:
        if os.environ.get(LIVE_KEY_VAR):
            raise LLMUnavailable(
                f"{LIVE_KEY_VAR} is set in this process. A research process runs without "
                "the live broker key at all, and this one is about to send text to a "
                f"third party; run it in an environment without {LIVE_KEY_VAR} "
                f"(for example `env -u {LIVE_KEY_VAR} tb research cycle ...`)."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailable(
                "the Anthropic SDK is not installed. `uv sync --extra llm` installs it; "
                "the deterministic searcher (`--proposer random`) needs nothing extra."
            ) from exc
        self._client = anthropic.Anthropic(timeout=timeout_seconds)
        self._model = model
        self._max_tokens = max_tokens
        self._fallbacks = fallbacks

    @property
    def model(self) -> str:
        return self._model

    def complete(self, *, system: str, user: str) -> LLMResponse:
        import anthropic
        from anthropic.types.beta import BetaTextBlock

        try:
            with self._client.beta.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                betas=[FALLBACK_BETA] if self._fallbacks else anthropic.omit,
                fallbacks="default" if self._fallbacks else anthropic.omit,
            ) as stream:
                message = stream.get_final_message()
        # Status codes and not bodies in the credential cases: the body is the
        # server's text, and a message about a key is the one kind of message
        # that should not be echoed into a terminal or a log by habit.
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMUnavailable(
                f"the Claude API refused the credentials (HTTP {exc.status_code}). The SDK "
                "reads ANTHROPIC_API_KEY; check that it is set and allowed to use "
                f"{self._model}."
            ) from exc
        except anthropic.NotFoundError as exc:
            raise LLMUnavailable(
                f"the Claude API does not know the model {self._model!r} (HTTP 404)"
            ) from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(
                f"the Claude API rejected the request (HTTP 400): {exc.message}. If it "
                "names the fallback, re-run with `--no-fallback`."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(
                "the Claude API is rate limiting this key and the SDK's retries are spent; "
                "try again later"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(
                f"the Claude API returned HTTP {exc.status_code}: {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the Claude API: {exc}") from exc
        except TypeError as exc:
            # What the SDK raises when no credential resolves at all — before
            # any request is made. Every argument above is checked against the
            # SDK's own signature, so this is the credential case in practice.
            raise LLMUnavailable(
                "no Claude API credential is configured. Set ANTHROPIC_API_KEY in this "
                f"process's environment ({exc})."
            ) from exc

        text = "".join(block.text for block in message.content if isinstance(block, BetaTextBlock))
        iterations = message.usage.iterations or []
        refusal = ""
        if message.stop_reason == "refusal" and message.stop_details is not None:
            details = message.stop_details
            refusal = ": ".join(part for part in (details.category, details.explanation) if part)
        return LLMResponse(
            text=text,
            requested_model=self._model,
            served_model=str(message.model),
            stop_reason=message.stop_reason,
            fell_back=any(entry.type == "fallback_message" for entry in iterations),
            refusal=refusal,
        )


# --------------------------------------------------------------------------
# Reading the reply
# --------------------------------------------------------------------------

# The start of an array of objects. Searched for rather than assuming the reply
# starts with one, because models wrap JSON in fences and prose despite being
# told not to — and "[3]" in a sentence must not be mistaken for the payload.
_ARRAY_OF_OBJECTS = re.compile(r"\[\s*\{")
_JSON_SPACE = re.compile(r"[ \t\n\r]*")
_CUT_OFF = "the reply ends inside the array: it was cut off, usually by the output budget"


def extract_items(text: str, *, limit: int) -> tuple[list[object], str]:
    """The values of the first JSON array of objects in `text`, and why reading stopped.

    One value at a time with `raw_decode`, rather than `json.loads` over a
    guessed slice, so a reply cut off by the output budget still yields every
    item that was complete — and so trailing prose after the array cannot make
    the whole reply unparseable. Returns at most `limit` values; the second
    element is empty on a clean read and otherwise says what went wrong, for
    the report.

    Parsing only. Every number is a `Decimal` (`NaN` and `Infinity` included,
    for the schema to refuse), and a value nested past the parser's recursion
    limit is a stop, not a crash.
    """
    text = text[:MAX_REPLY_CHARS]
    match = _ARRAY_OF_OBJECTS.search(text)
    if match is None:
        return [], "the reply holds no JSON array of objects"
    decoder = json.JSONDecoder(parse_float=Decimal, parse_constant=Decimal)
    items: list[object] = []
    position = match.start() + 1
    while True:
        position = _skip_space(text, position)
        if position >= len(text):
            return items, _CUT_OFF
        if text[position] == "]":
            return items, ""
        if len(items) >= limit:
            return items, (
                f"the reply held more than the {limit} item(s) asked for; the rest were not read"
            )
        try:
            item, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            # An error at the very end, or a string that never closes, can only
            # mean the text stopped mid-item: a string runs to its closing quote,
            # so an unterminated one ran off the end of the reply.
            if exc.pos >= len(text) or exc.msg.startswith("Unterminated string"):
                return items, _CUT_OFF
            return items, _tidy(f"item {len(items)} is not valid JSON: {exc}")
        except (ValueError, RecursionError) as exc:
            return items, _tidy(f"item {len(items)} is not valid JSON: {type(exc).__name__}: {exc}")
        items.append(item)
        position = _skip_space(text, position)
        if position < len(text) and text[position] == ",":
            position += 1
        elif position < len(text) and text[position] != "]":
            return items, f"expected ',' or ']' after item {len(items) - 1}"


def _skip_space(text: str, position: int) -> int:
    matched = _JSON_SPACE.match(text, position)
    return position if matched is None else matched.end()


def _tidy(text: str, limit: int = MAX_REASON_CHARS) -> str:
    """Model-derived text made safe to print and to ledger.

    Refusal reasons quote the model's own values back (the schema's errors
    include the offending input), and a key the model invented is the model's
    text verbatim. Both reach a terminal and the event log, so anything
    unprintable becomes `?` and the length is bounded. Two cases in particular:
    an ANSI escape can rewrite what an operator sees, and a lone surrogate —
    which JSON can spell and UTF-8 cannot encode — would make the ledger append
    raise.
    """
    printable = "".join(ch if ch.isprintable() else "?" for ch in text)
    return printable if len(printable) <= limit else printable[: limit - 1] + "…"


# --------------------------------------------------------------------------
# The proposer
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LLMReport:
    """One call, as the cycle records it and the operator reads it."""

    requested_model: str
    served_model: str
    fell_back: bool
    stop_reason: str | None
    n_requested: int
    n_items: int
    accepted: tuple[Proposal, ...]
    n_refused: int
    refused: tuple[str, ...]
    n_duplicates: int
    ignored_keys: tuple[str, ...]
    stopped: str
    system_prompt: str
    user_prompt: str
    reply_sha256: str
    reply_chars: int

    def lines(self) -> tuple[str, ...]:
        served = self.served_model
        if self.fell_back:
            served = f"{self.served_model}, after {self.requested_model} declined"
        out = [
            f"asked {served} for {self.n_requested} spec(s): {self.n_items} item(s) read, "
            f"{len(self.accepted)} accepted, {self.n_refused} refused as malformed, "
            f"{self.n_duplicates} duplicate(s) dropped"
        ]
        if self.stopped:
            out.append(f"reading stopped early: {self.stopped}")
        if self.stop_reason == "max_tokens":
            out.append(
                "the reply used its whole output budget; fewer specs per call leaves room "
                "for every one to finish"
            )
        if self.ignored_keys:
            out.append("ignored keys the model has no say over: " + ", ".join(self.ignored_keys))
        out.extend(f"refused: {reason}" for reason in self.refused[:3])
        out.append(
            "a model is not reproducible from the seed; the prompts, a hash of the reply "
            "and every accepted spec are recorded instead"
        )
        return tuple(out)


class LLMProposer:
    """Asks a model for specs, one call per `propose`, and keeps what validates.

    Built per search like the other proposers, because the prompt depends on
    what that search may use: the lookback ladder the training window supports,
    the edge band the limits allow, and the budget whose haircut the model is
    told about. `reports` holds one `LLMReport` per call, for the cycle to
    ledger.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        bounds: ProposalBounds,
        regime: RegimeDescription | None = None,
        required_sharpe: float | None = None,
        n_trials: int | None = None,
    ) -> None:
        self._client = client
        self._bounds = bounds
        self._regime = regime if regime is not None else RegimeDescription()
        self._required_sharpe = required_sharpe
        self._n_trials = n_trials
        self.reports: list[LLMReport] = []

    @property
    def name(self) -> str:
        return "llm"

    def propose(
        self,
        *,
        n: int,
        rng: random.Random,
        parents: Sequence[StrategySpec] = (),
    ) -> list[Proposal]:
        """One call; every accepted item as a proposal.

        `rng` and `parents` are part of the interface and unused here. The model
        is not seeded by the search, and it is not shown parents: a parent is a
        spec selected on training performance, and handing the selection back to
        the model would make it a second, unrecorded optimiser over the same
        window.
        """
        requested = min(n, MAX_SPECS_PER_CALL)
        if requested < 1:
            return []
        system = render_system_prompt(self._bounds)
        user = render_user_prompt(
            n=requested,
            regime=self._regime,
            required_sharpe=self._required_sharpe,
            n_trials=self._n_trials,
        )
        # Both halves, every call. The tripwire behind the types — see
        # `prompts` for why the two are layered.
        audit_prompt(system)
        audit_prompt(user)

        response = self._client.complete(system=system, user=user)
        if response.stop_reason == "refusal":
            raise LLMError(
                f"{_tidy(response.served_model, 80)} declined the request"
                + (f" ({_tidy(response.refusal)})" if response.refusal else "")
                + ". Nothing was evaluated or recorded; `--proposer random` runs the same "
                "search without a model."
            )

        items, stopped = extract_items(response.text, limit=requested)
        proposals: list[Proposal] = []
        refused: list[str] = []
        ignored: set[str] = set()
        seen_trees: set[str] = set()
        n_refused = 0
        n_duplicates = 0
        served = _tidy(response.served_model, 80)
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                n_refused += 1
                refused.append(f"item {index} is a JSON {type(item).__name__}, not an object")
                continue
            ignored.update(_tidy(str(key), 40) for key in item if key not in MODEL_KEYS)
            payload: dict[str, object] = {key: item[key] for key in MODEL_KEYS if key in item}
            payload["name"] = f"llm-{index:04d}"
            payload["notes"] = f"proposed by a language model ({served})"
            payload["min_holding_minutes"] = self._bounds.min_holding_minutes
            try:
                spec = StrategySpec.parse(payload)
            except SpecError as exc:
                n_refused += 1
                refused.append(_tidy(f"item {index}: {exc}"))
                continue
            dumped = spec.model_dump(mode="json")
            tree = hash_payload({key: dumped[key] for key in sorted(MODEL_KEYS)})
            if tree in seen_trees:
                # The same idea twice in one reply. Evaluating it twice would
                # spend two trials on one result.
                n_duplicates += 1
                continue
            seen_trees.add(tree)
            proposals.append(
                Proposal(
                    spec=spec,
                    author_kind=AuthorKind.LLM,
                    operator="llm",
                    detail=f"item {index} of a reply from {served}",
                )
            )

        report = LLMReport(
            requested_model=response.requested_model,
            served_model=served,
            fell_back=response.fell_back,
            stop_reason=response.stop_reason,
            n_requested=requested,
            n_items=len(items),
            accepted=tuple(proposals),
            n_refused=n_refused,
            refused=tuple(refused[:MAX_REASONS_KEPT]),
            n_duplicates=n_duplicates,
            ignored_keys=tuple(sorted(ignored))[:MAX_REASONS_KEPT],
            stopped=stopped,
            system_prompt=system,
            user_prompt=user,
            # `surrogatepass` because JSON can spell a lone surrogate ("\ud800")
            # that strict UTF-8 cannot encode, and a hash that raised on the
            # reply would turn one hostile string into a crashed search.
            reply_sha256=sha256_hex(response.text.encode("utf-8", errors="surrogatepass")),
            reply_chars=len(response.text),
        )
        self.reports.append(report)
        if not proposals:
            why = stopped or (refused[0] if refused else "every item was a duplicate")
            raise LLMError(
                f"{served} returned no usable spec ({why}). Nothing was evaluated or recorded."
            )
        return proposals
