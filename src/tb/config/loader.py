"""Load the hard limits and pin their hash.

Two hashes are recorded, because they answer different questions:

* `content_sha256` is over the file's raw bytes. It is the tamper pin, and it
  moves if a comment or a space changes — which is what you want from a pin.
* `canonical_sha256` is over the parsed, validated, canonicalised values. It
  answers "did the limits actually change", so a reformat is distinguishable
  from a loosened cap in the audit trail.

The pin is re-verified on every cycle. A mismatch is a halt, not a reload: the
process validated its behaviour against the limits it read at startup, and
silently adopting different ones mid-flight is how a bot ends up trading under
caps nobody reviewed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from tb.config.hard_limits import HardLimits
from tb.core.canonical import hash_file, hash_payload
from tb.core.clock import now_utc
from tb.core.errors import ConfigDriftError, ConfigError

DEFAULT_PATH = Path("config/hard_limits.yaml")
ENV_PATH_VAR = "TB_HARD_LIMITS_PATH"

# The schema versions this build knows how to interpret. A limits file from the
# future is refused rather than partially understood.
SUPPORTED_SCHEMA_VERSIONS = frozenset({2})


@dataclass(frozen=True, slots=True)
class ImmutabilityReport:
    """Whether the limits file is genuinely beyond this process's reach.

    The design claims the agent cannot modify its own constraints. That claim is
    only true if the filesystem enforces it, so it is measured and reported
    rather than assumed. `tb doctor` surfaces this.
    """

    path: Path
    writable_by_process: bool
    running_as_root: bool
    file_owner_uid: int
    process_uid: int

    @property
    def enforced(self) -> bool:
        """True only if the kernel would actually refuse our writes."""
        return not self.writable_by_process and not self.running_as_root

    @property
    def explanation(self) -> str:
        if self.running_as_root:
            return (
                "process is running as root, so file permissions cannot constrain it. "
                "Run the bot as a non-root user and mount config/ read-only for that user."
            )
        if self.writable_by_process:
            owner = (
                "the same user that runs the bot"
                if self.file_owner_uid == self.process_uid
                else f"uid {self.file_owner_uid}"
            )
            return (
                f"limits file is writable by this process (owned by {owner}). "
                "Mount it read-only, or chown it to a different user, so the "
                "constraint is enforced by the kernel rather than by convention."
            )
        return "limits file is not writable by this process; the constraint is enforced."


@dataclass(frozen=True, slots=True)
class PinnedLimits:
    """Validated hard limits, together with the pin that detects drift."""

    limits: HardLimits
    source_path: Path
    content_sha256: str
    canonical_sha256: str
    loaded_at: datetime
    immutability: ImmutabilityReport

    @property
    def config_hash(self) -> str:
        """The value written to `event_log.config_hash` on every event."""
        return self.content_sha256

    def verify_unchanged(self) -> None:
        """Re-hash the file on disk; raise if it no longer matches the pin.

        Called every cycle. Deliberately cheap: one SHA-256 over a small file.
        """
        try:
            current = hash_file(str(self.source_path))
        except OSError as exc:
            # Unreadable limits are as disqualifying as changed limits: we can no
            # longer demonstrate we are operating under the reviewed caps.
            raise ConfigDriftError(
                f"hard limits at {self.source_path} became unreadable mid-run: {exc}"
            ) from exc

        if current != self.content_sha256:
            raise ConfigDriftError(
                f"hard limits at {self.source_path} changed while running "
                f"(pinned {self.content_sha256[:12]}…, found {current[:12]}…). "
                "Restart to adopt them; they will not be applied mid-run."
            )

    def audit_record(self) -> dict[str, Any]:
        """The projection recorded in `config_versions` at startup."""
        return {
            "config_hash": self.content_sha256,
            "canonical_hash": self.canonical_sha256,
            "kind": "hard_limits",
            "source_path": str(self.source_path),
            "schema_version": self.limits.schema_version,
            "currency": self.limits.currency,
            "immutability_enforced": self.immutability.enforced,
            "immutability_detail": self.immutability.explanation,
            "loaded_at": self.loaded_at,
            "values": self.limits.model_dump(mode="json"),
        }


def resolve_path(path: str | Path | None = None) -> Path:
    """Pick the limits path: explicit argument, then env var, then the default."""
    if path is not None:
        return Path(path)
    from_env = os.environ.get(ENV_PATH_VAR)
    if from_env:
        return Path(from_env)
    return DEFAULT_PATH


def _assess_immutability(path: Path) -> ImmutabilityReport:
    process_uid = os.getuid()
    try:
        owner_uid = path.stat().st_uid
    except OSError:
        owner_uid = -1
    return ImmutabilityReport(
        path=path,
        # os.access reflects the real uid's permissions. It reports True for
        # root regardless of mode bits, which is why root is tracked separately.
        writable_by_process=os.access(path, os.W_OK),
        running_as_root=(process_uid == 0),
        file_owner_uid=owner_uid,
        process_uid=process_uid,
    )


def load_hard_limits(path: str | Path | None = None) -> PinnedLimits:
    """Read, validate, and pin the hard limits.

    Every failure mode here is fatal by design. There is no default set of
    limits to fall back on: a trading bot with no enforced caps must not start.
    """
    resolved = resolve_path(path)

    try:
        raw = resolved.read_bytes()
    except FileNotFoundError as exc:
        raise ConfigError(
            f"no hard limits at {resolved}. The bot will not start without them — "
            "copy config/hard_limits.yaml from the repository and review the values."
        ) from exc
    except OSError as exc:
        raise ConfigError(f"cannot read hard limits at {resolved}: {exc}") from exc

    content_sha256 = hash_file(str(resolved))

    try:
        # safe_load, never load: this file is trusted input, but a YAML loader
        # that can construct arbitrary Python objects has no business in the
        # startup path of a system that handles money.
        parsed = yaml.safe_load(raw.decode("utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ConfigError(f"hard limits at {resolved} are not valid YAML: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ConfigError(
            f"hard limits at {resolved} must be a YAML mapping, got {type(parsed).__name__}"
        )

    version = parsed.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ConfigError(
            f"hard limits at {resolved} declare schema_version={version!r}; "
            f"this build supports {sorted(SUPPORTED_SCHEMA_VERSIONS)}. Refusing to "
            "guess at the meaning of an unknown limits schema."
        )

    try:
        limits = HardLimits.model_validate(parsed)
    except Exception as exc:
        # Pydantic's message names the offending field and why. Surfacing it
        # verbatim is more useful than any summary we could write.
        raise ConfigError(f"hard limits at {resolved} are invalid:\n{exc}") from exc

    return PinnedLimits(
        limits=limits,
        source_path=resolved,
        content_sha256=content_sha256,
        canonical_sha256=hash_payload(limits.model_dump(mode="json")),
        loaded_at=now_utc(),
        immutability=_assess_immutability(resolved),
    )
