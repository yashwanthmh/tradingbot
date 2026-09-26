// The tb dashboard: reads the ledger's views and draws them.
//
// Every string that reaches the page goes in as a text node. Tickers, halt
// details, model output recorded on a proposal — all of it came from outside
// at some point, and none of it is ever parsed as markup here. The server's
// content policy forbids inline script as a second lock on the same door.
"use strict";

(() => {
  const TOKEN_KEY = "tb.dashboard.token";
  const MODE_KEY = "tb.dashboard.mode";
  const REFRESH_MS = 10000;
  const MAX_EVENTS = 200;
  const SVG_NS = "http://www.w3.org/2000/svg";

  const state = {
    mode: stored(MODE_KEY) || "demo",
    after: 0,
    events: [],
    buttonEnabled: false,
    timer: null,
  };

  // -- storage: per tab, and never required ----------------------------------

  function stored(key) {
    try {
      return window.sessionStorage.getItem(key);
    } catch (_) {
      return null;
    }
  }

  function store(key, value) {
    try {
      if (value === null) {
        window.sessionStorage.removeItem(key);
      } else {
        window.sessionStorage.setItem(key, value);
      }
    } catch (_) {
      // A private window may refuse storage; the page still works for this visit.
    }
  }

  // -- building nodes ---------------------------------------------------------

  function el(tag, className, ...children) {
    const node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    for (const child of children) {
      if (child === null || child === undefined) {
        continue;
      }
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function svg(tag, attributes) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attributes || {})) {
      node.setAttribute(name, String(value));
    }
    return node;
  }

  function show(id, ...nodes) {
    document.getElementById(id).replaceChildren(...nodes);
  }

  function table(headings, rows) {
    const head = el("tr", null, ...headings.map((h) => el("th", null, h)));
    const body = rows.map((cells) =>
      el("tr", null, ...cells.map((c) => (c instanceof Node && c.tagName === "TD" ? c : el("td", null, c)))),
    );
    return el("table", null, el("thead", null, head), el("tbody", null, ...body));
  }

  function badge(text, kind) {
    return el("span", `badge ${kind || ""}`.trim(), text);
  }

  function muted(text) {
    return el("p", "muted", text);
  }

  // -- formatting -------------------------------------------------------------

  function pct(value) {
    return typeof value === "number" ? `${value.toFixed(2)}%` : "—";
  }

  function money(value, currency) {
    return value === null || value === undefined ? "—" : `${value} ${currency || ""}`.trim();
  }

  function when(iso) {
    if (!iso) {
      return "—";
    }
    const at = new Date(iso);
    return Number.isNaN(at.getTime()) ? String(iso) : at.toLocaleString();
  }

  function measure(value, unit) {
    if (value === null || value === undefined) {
      return "unknown";
    }
    if (typeof value === "number") {
      return unit.startsWith("%") ? `${value.toFixed(2)}${unit}` : `${value} ${unit}`;
    }
    return `${value} ${unit}`;
  }

  // -- talking to the server --------------------------------------------------

  class ApiError extends Error {
    constructor(status, message) {
      super(message);
      this.status = status;
    }
  }

  async function api(path, options) {
    const init = Object.assign({ cache: "no-store", credentials: "omit" }, options || {});
    const headers = new Headers(init.headers || {});
    const token = stored(TOKEN_KEY);
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
    init.headers = headers;
    const response = await fetch(path, init);
    const type = response.headers.get("content-type") || "";
    const body = type.includes("application/json") ? await response.json() : await response.text();
    // Only an answer to the token in use now says anything about it: a 401
    // to a request sent before the token was entered must not ask again.
    if (token === stored(TOKEN_KEY)) {
      if (response.status === 401) {
        askForToken(true);
      } else if (response.ok) {
        askForToken(false);
      }
    }
    if (!response.ok) {
      const detail = body && typeof body === "object" && "detail" in body ? body.detail : body;
      throw new ApiError(response.status, typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return body;
  }

  function askForToken(needed) {
    document.getElementById("token-form").hidden = !needed;
    document.getElementById("forget-token").hidden = needed || !stored(TOKEN_KEY);
  }

  // -- panels -----------------------------------------------------------------

  function renderStatus(s) {
    const run = s.run_state;
    const running = s.running
      ? `${s.running.mode || "unknown mode"} run ${s.running.run_id} (pid ${s.running.pid} on ${s.running.host})`
      : "no loop holds the lease";
    const heartbeat = s.heartbeat.stale
      ? badge("stale", "bad")
      : badge(`${Math.round(s.heartbeat.age_seconds || 0)}s old`, "good");
    const live = s.live.armed
      ? badge(`armed: ${s.live.strategies.join(", ")}`, "warn")
      : badge(s.live.enabled_in_limits ? "enabled in limits, not armed" : "disabled", "quiet");
    const rows = [
      ["Run state", el("span", null, badge(run.state, `state-${run.state}`), ` since ${when(run.since)}`)],
      ["Reason", run.reason || "—"],
      ["Kill switch", badge(s.kill_switch.state, s.kill_switch.may_trade ? "good" : "bad")],
      ["Heartbeat", heartbeat],
      ["Loop", running],
      ["Live", el("span", null, live, s.live.armed ? "" : ` ${s.live.reason || ""}`)],
      ["Ledger head", s.ledger.head_seq === null ? "empty" : `#${s.ledger.head_seq}`],
      ["Limits", `${s.limits.config_hash.slice(0, 12)}… (${s.limits.currency})`],
    ];
    const list = el("dl", "facts");
    for (const [term, value] of rows) {
      list.append(el("dt", null, term), el("dd", null, value));
    }
    const parts = [list];
    if (s.limits.drift) {
      parts.push(el("p", "alert", `Limits changed on disk: ${s.limits.drift}`));
    }
    if (s.open_halts.length) {
      parts.push(el("h3", null, `Open halts (${s.open_halts.length})`));
      parts.push(
        table(
          ["Raised", "Trigger", "Detail"],
          s.open_halts.map((h) => [when(h.raised_at), h.trigger, h.detail]),
        ),
      );
    }
    show("status", ...parts);
    document.getElementById("as-of").textContent = `as of ${when(s.as_of)}`;
    // A ledger shorter than the tail's cursor is a different ledger (a
    // restore, a fresh start): start the tail again rather than wait forever.
    if (s.ledger.head_seq !== null && s.ledger.head_seq < state.after) {
      state.after = 0;
      state.events = [];
    }

    state.buttonEnabled = Boolean(s.control && s.control.kill_switch_button);
    const button = document.getElementById("kill-button");
    button.disabled = !state.buttonEnabled;
    button.title = state.buttonEnabled
      ? ""
      : "Start the dashboard with TB_DASHBOARD_TOKEN set to enable this button; tb halt works without it.";
    if (!state.buttonEnabled && !document.getElementById("kill-note").textContent) {
      document.getElementById("kill-note").textContent =
        "The button is off: start the dashboard with TB_DASHBOARD_TOKEN set. `tb halt --reason …` works without it.";
    }
  }

  function chart(points) {
    const W = 720;
    const H = 220;
    const PAD = 44;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": "equity curve" });
    const label = (x, y, text, anchor) => {
      const node = svg("text", { x, y, "text-anchor": anchor || "start" });
      node.textContent = text;
      return node;
    };
    if (points.length < 2) {
      root.append(label(W / 2, H / 2, points.length ? "one mark so far" : "no marks yet", "middle"));
      return root;
    }
    const xs = points.map((p) => Date.parse(p.at));
    const ys = points.map((p) => Number(p.equity));
    let lo = Math.min(...ys);
    let hi = Math.max(...ys);
    if (hi === lo) {
      hi += 1;
      lo -= 1;
    }
    const x0 = xs[0];
    const x1 = xs[xs.length - 1] === x0 ? x0 + 1 : xs[xs.length - 1];
    const sx = (t) => PAD + ((t - x0) / (x1 - x0)) * (W - 2 * PAD);
    const sy = (v) => H - PAD - ((v - lo) / (hi - lo)) * (H - 2 * PAD);
    root.append(svg("line", { x1: PAD, y1: H - PAD, x2: W - PAD, y2: H - PAD, class: "axis" }));
    root.append(svg("line", { x1: PAD, y1: PAD, x2: PAD, y2: H - PAD, class: "axis" }));
    root.append(
      svg("polyline", {
        class: "line",
        points: xs.map((t, i) => `${sx(t).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(" "),
      }),
    );
    root.append(label(PAD - 6, sy(hi) + 4, points[ys.indexOf(Math.max(...ys))].equity, "end"));
    root.append(label(PAD - 6, sy(lo) + 4, points[ys.indexOf(Math.min(...ys))].equity, "end"));
    root.append(label(PAD, H - PAD + 18, when(points[0].at)));
    root.append(label(W - PAD, H - PAD + 18, when(points[points.length - 1].at), "end"));
    return root;
  }

  function renderEquity(e) {
    if (!e.run_id) {
      show("equity", muted(`No ${e.mode} run has been recorded yet.`));
      return;
    }
    const r = e.reading;
    const facts = el("dl", "facts inline");
    for (const [term, value] of [
      ["Equity", money(r.equity, e.currency)],
      ["Peak", money(r.peak, e.currency)],
      ["Today", pct(r.day_pnl_pct)],
      ["Five sessions", pct(r.rolling_pnl_pct)],
      ["From peak", pct(r.drawdown_from_peak_pct === null ? null : -r.drawdown_from_peak_pct)],
    ]) {
      facts.append(el("dt", null, term), el("dd", null, value));
    }
    const parts = [chart(e.points), facts];
    for (const caveat of r.caveats) {
      parts.push(el("p", "caveat", caveat));
    }
    show("equity", ...parts);
  }

  function meter(used) {
    const bar = el("span", "meter");
    const fill = el("span", "fill");
    const share = typeof used === "number" ? Math.max(0, Math.min(100, used)) : 0;
    fill.style.width = `${share}%`;
    if (used === null || used === undefined) {
      bar.classList.add("unknown");
    } else if (used >= 80) {
      bar.classList.add("hot");
    } else if (used >= 50) {
      bar.classList.add("warm");
    }
    bar.append(fill);
    return bar;
  }

  function renderRisk(r) {
    if (!r.budgets.length) {
      show("risk", muted("Nothing to measure yet."));
      return;
    }
    show(
      "risk",
      table(
        ["Budget", "Observed", "Limit", "Used"],
        r.budgets.map((b) => [
          b.name,
          measure(b.observed, b.unit),
          measure(b.limit, b.unit),
          el("td", null, meter(b.used_pct), typeof b.used_pct === "number" ? ` ${b.used_pct.toFixed(0)}%` : " —"),
        ]),
      ),
    );
  }

  function renderStrategies(rows) {
    if (!rows.length) {
      show("strategies", muted("No strategy has a standing or a closed trade yet."));
      return;
    }
    show(
      "strategies",
      table(
        ["Strategy", "Status", "Rung", "Closed here", "Measured", "Wins", "Realised here", "Lineage budget used"],
        rows.map((s) => [
          s.strategy_id === null ? "(unattributed)" : `${s.strategy_id} v${s.version}`,
          s.status === null ? "—" : badge(s.status, `status-${s.status}`),
          s.rung === null ? "—" : s.rung,
          s.account.trades,
          s.account.measured,
          s.account.wins,
          s.account.realised,
          s.lineage_budget === null ? "—" : `${s.lineage_consumed} of ${s.lineage_budget}`,
        ]),
      ),
    );
  }

  function renderSessions(s) {
    const streak = s.required
      ? `${s.streak} of ${s.required} clean sessions in a row`
      : `${s.streak} clean sessions in a row`;
    const head = el("p", null, streak, `, ${s.trades_in_streak} closed trades among them.`);
    if (!s.sessions.length) {
      show("sessions", head, muted(`No ${s.mode} session on record.`));
      return;
    }
    show(
      "sessions",
      head,
      table(
        ["Session", "Verdict", "Coverage", "Cycles", "Orders", "Fills", "Trades", "Why"],
        s.sessions.map((x) => [
          x.date,
          badge(x.verdict, `verdict-${x.verdict}`),
          `${x.coverage_pct}%`,
          x.cycles,
          x.orders,
          x.fills,
          x.trades,
          x.why,
        ]),
      ),
    );
  }

  function renderEvents() {
    if (!state.events.length) {
      show("events", muted("No events yet."));
      return;
    }
    const list = el("ol", "events");
    for (const e of [...state.events].reverse()) {
      const summary = el(
        "summary",
        null,
        el("span", "seq", `#${e.seq}`),
        el("span", "ts", when(e.ts)),
        el("span", "type", e.type),
        el("span", "who", e.actor),
        el("span", "agg", e.aggregate_id),
      );
      const detail = el(
        "pre",
        "payload",
        e.truncated ? "(payload too long to carry here: read it with tb ledger)" : JSON.stringify(e.payload, null, 2),
      );
      list.append(el("li", null, el("details", null, summary, detail)));
    }
    show("events", list);
  }

  function renderJournal(pages) {
    if (!pages.length) {
      show("journal", muted("No journal pages yet: tb journal write makes them."));
      return;
    }
    const list = el("ul", "pages");
    for (const page of pages) {
      const button = el("button", "link", page.date);
      button.type = "button";
      button.addEventListener("click", () => openPage(page.date));
      list.append(el("li", null, button, " ", badge(page.status || "unreadable", `page-${page.status || "bad"}`)));
    }
    show("journal", list);
  }

  async function openPage(day) {
    const view = document.getElementById("journal-page");
    try {
      view.textContent = await api(`/api/journal/${encodeURIComponent(day)}`);
    } catch (err) {
      view.textContent = `Could not load ${day}: ${err.message}`;
    }
    view.hidden = false;
  }

  // -- refreshing -------------------------------------------------------------

  async function panel(id, path, render) {
    try {
      render(await api(path));
    } catch (err) {
      show(id, el("p", "alert", err.status === 401 ? "Needs the dashboard token." : `Unavailable: ${err.message}`));
    }
  }

  async function tail() {
    try {
      const fresh = await api(`/api/events?after=${state.after}&limit=100`);
      if (fresh.length) {
        state.events = state.events.concat(fresh).slice(-MAX_EVENTS);
        state.after = fresh[fresh.length - 1].seq;
      }
      renderEvents();
    } catch (err) {
      show("events", el("p", "alert", `Unavailable: ${err.message}`));
    }
  }

  async function refresh() {
    const mode = encodeURIComponent(state.mode);
    await Promise.allSettled([
      panel("status", "/api/status", renderStatus),
      panel("equity", `/api/equity?mode=${mode}`, renderEquity),
      panel("risk", `/api/risk?mode=${mode}`, renderRisk),
      panel("strategies", `/api/strategies?mode=${mode}`, renderStrategies),
      panel("sessions", `/api/sessions?mode=${mode}`, renderSessions),
      panel("journal", "/api/journal", renderJournal),
      tail(),
    ]);
  }

  function schedule() {
    window.clearInterval(state.timer);
    state.timer = document.hidden ? null : window.setInterval(refresh, REFRESH_MS);
  }

  // -- the kill switch --------------------------------------------------------

  async function engage(event) {
    event.preventDefault();
    const note = document.getElementById("kill-note");
    const reason = document.getElementById("kill-reason").value.trim();
    if (!reason) {
      note.textContent = "Say why you are stopping it: the reason goes on the record.";
      return;
    }
    const sure = window.confirm(
      "Engage the kill switch?\n\nNew trading stops at the next check. Releasing it needs `tb resume` at a terminal.",
    );
    if (!sure) {
      return;
    }
    const button = document.getElementById("kill-button");
    button.disabled = true;
    try {
      const result = await api("/api/killswitch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason }),
      });
      if (result.already_engaged) {
        note.textContent = "Already engaged. Nothing new was recorded.";
      } else if (result.recorded) {
        note.textContent = `Engaged. Halt ${result.halt_id} is on the record.`;
      } else {
        note.textContent = `Engaged. ${result.problem}`;
      }
    } catch (err) {
      note.textContent = `NOT engaged: ${err.message}. Run tb halt --reason "…" now.`;
    } finally {
      button.disabled = !state.buttonEnabled;
      refresh();
    }
  }

  // -- wiring -----------------------------------------------------------------

  function start() {
    const select = document.getElementById("mode");
    select.value = state.mode;
    select.addEventListener("change", () => {
      state.mode = select.value;
      store(MODE_KEY, state.mode);
      refresh();
    });
    document.getElementById("token-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const input = document.getElementById("token");
      const token = input.value.trim();
      input.value = "";
      store(TOKEN_KEY, token || null);
      askForToken(false);
      refresh();
    });
    document.getElementById("forget-token").addEventListener("click", () => {
      store(TOKEN_KEY, null);
      askForToken(true);
    });
    document.getElementById("kill-form").addEventListener("submit", engage);
    document.addEventListener("visibilitychange", () => {
      schedule();
      if (!document.hidden) {
        refresh();
      }
    });
    askForToken(false);
    refresh();
    schedule();
  }

  start();
})();
