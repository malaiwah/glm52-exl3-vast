#!/usr/bin/env python3
"""vast.ai "Open" button landing page — dual-protocol (TLS + plain) on :1111.

Started first thing in entrypoint.sh so status is visible from the very first
boot minute. Each connection's first byte is peeked: 0x16 = TLS handshake,
else plain HTTP. Once a cert exists (lazy-loaded mid-boot), plain hits are
302-redirected to the HTTPS view. Boot state comes from the status file the
entrypoint rewrites at each milestone, plus live probes (download size,
/health). The API key, filled-in client configs (oh-my-pi, opencode, Claude
Code, Codex), live metrics dashboard and /chat render only over TLS with a
validated OPEN_BUTTON_TOKEN — or over plain HTTP if LANDING_ALLOW_INSECURE=1
(trusted-LAN/demo hosts without a cert). The dashboard scrapes vLLM's
/metrics from the browser (vLLM CORS is open) — no server-side proxying.
Requires OPEN_BUTTON_PORT=1111 env + '-p 1111:1111' in the template.
"""
import hmac
import html
import json
import os
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from string import Template
from urllib.parse import parse_qs, urlparse

# The knob registry, the inheritance model and the validation matrix live in
# scripts/glm_config.py so that the entrypoint and this page cannot disagree
# about them. If it cannot be imported the page still serves — it just loses
# the config editor rather than the whole landing page.
SCRIPTS_DIR = os.environ.get("GLM_SCRIPTS_DIR", "/opt/scripts")
sys.path.insert(0, SCRIPTS_DIR)
try:
    import glm_config as gc
except Exception as _e:  # pragma: no cover - import guard
    gc = None
    _GC_ERR = str(_e)
try:
    import provider as prov
    import terminate_worker
except Exception:        # the terminate control simply does not appear
    prov = None
    terminate_worker = None
try:
    import soul_config as soul
except Exception as _e:  # SOUL is optional and defaults to disabled
    soul = None
    _SOUL_ERR = str(_e)

TOKEN = os.environ.get("OPEN_BUTTON_TOKEN", "")
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/GLM-5.2-EXL3-TR3-3.0bpw")
STATUS_FILE = os.environ.get("STATUS_FILE", "/tmp/glm-boot-status.json")
ALLOW_INSECURE = os.environ.get("LANDING_ALLOW_INSECURE", "0") == "1"
TRUST_PROXY_HTTPS = os.environ.get("LANDING_TRUST_PROXY_HTTPS", "0") == "1"
API_PORT = os.environ.get("LANDING_API_PORT", "8000")

# Per-connection socket timeout (seconds). A stalled client must not hold the
# single-threaded accept loop or a handler thread indefinitely.
LANDING_TIMEOUT = float(os.environ.get("LANDING_TIMEOUT", "30"))

# Serializes the check-then-spawn sequence in start_termination so two
# concurrent POSTs cannot both pass the "already in progress" check.
_terminate_lock = threading.Lock()


_ssl_ctx = None


def is_secure_connection(connection) -> bool:
    """Runpod's managed proxy terminates TLS before forwarding to this port."""
    return isinstance(connection, ssl.SSLSocket) or ALLOW_INSECURE or TRUST_PROXY_HTTPS


def ssl_ctx():
    """Lazy: the cert is issued minutes into the boot; pick it up when it lands."""
    global _ssl_ctx
    if _ssl_ctx is None:
        st = status()
        crt, key = st.get("cert", ""), st.get("keyfile", "")
        if crt and key and os.path.isfile(crt):
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            # The page carries capability and API tokens. Set an explicit
            # protocol floor instead of inheriting a potentially permissive
            # OpenSSL policy from the rental host.
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.load_cert_chain(crt, key)
            _ssl_ctx = ctx
    return _ssl_ctx


def status() -> dict:
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def soul_status(sanitized=False) -> dict:
    """SOUL status with config-derived defaults when its process is off."""
    if soul is None:
        return {"state": "Error", "level": 0, "controllerHealthy": False,
                "latestHeadline": "SOUL support is unavailable"}
    config, _sources, notes = soul.resolve()
    doc = soul.read_json(soul.paths()["status"], {}) or {}
    doc.update({
        "level": config["autonomyLevel"],
        "requestedLevel": config["requestedLevel"],
        "maximumLevel": config["maximumLevel"],
        "config": config,
        "configNotes": notes,
    })
    for key, default in (("lastRunAt", None), ("nextRunAt", None),
                         ("pendingIncidents", 0), ("controllerHealthy", False)):
        doc.setdefault(key, default)
    if config["autonomyLevel"] == 0:
        doc.update({"state": "Disabled", "controllerHealthy": False,
                    "latestHeadline": doc.get("latestHeadline", "")})
    elif not doc.get("state"):
        doc["state"] = "Waiting for model"
    pid = doc.get("controllerPid")
    if config["autonomyLevel"] > 0 and pid:
        try:
            os.kill(int(pid), 0)
        except (OSError, ValueError):
            doc.update({"state": "Error", "controllerHealthy": False})
    if sanitized:
        return {
            "state": str(doc.get("state", "Disabled"))[:80],
            "level": int(doc.get("level", 0)),
            "controllerHealthy": bool(doc.get("controllerHealthy", False)),
            "latestHeadline": str(soul.redact(
                str(doc.get("latestHeadline", ""))))[:160],
        }
    return soul.redact(doc)


def render_soul_card(tok="") -> str:
    doc = soul_status(sanitized=True)
    state = doc["state"]
    ok = state in ("Observing", "Investigating")
    level = doc["level"]
    headline = doc.get("latestHeadline") or (
        "Proactive diagnostics are off." if level == 0 else
        "The SOUL is waiting for the verified local model.")
    link = ""
    if TOKEN and tok:
        link = (f' <a href="/soul?token={html.escape(tok, quote=True)}">'
                "Open journal &amp; settings &rarr;</a>")
    return (
        "<div class=card><h3>Appliance SOUL</h3>"
        f"<div class=v><span class='pill {'ok' if ok else 'busy'}'></span>"
        f"<span class={'ok' if ok else 'busy'}>{html.escape(state)}</span>"
        f" <span class=sub>level {level}</span></div>"
        f"<div class=sub style='margin-top:.45rem'>{html.escape(headline)}{link}</div>"
        "</div>")


def _soul_evidence(ref):
    if soul is None or not isinstance(ref, str):
        return None
    root = soul.paths()["root"].resolve()
    try:
        target = (root / ref).resolve()
        if root not in target.parents or not target.is_file():
            return None
        return soul.redact(json.loads(target.read_text(encoding="utf-8")))  # type: ignore
    except (OSError, ValueError):
        return None


def render_soul(tok, secure, message="", cls="", before="") -> bytes:
    doc = soul_status()
    config = doc.get("config", {})
    entries = soul.read_journal(limit=20, before=before) if soul is not None else []
    tok_q = html.escape(tok, quote=True)
    parts = [page_head("Appliance SOUL"),
             "<div class='wrap'><header class=hero><h1><b>Appliance SOUL</b></h1>"
             "<span class=sub>A local diagnostic journal. Startup verification and "
             "rollback remain authoritative.</span></header>",
             f"<p><a href='/?token={tok_q}'>&larr; Appliance</a></p>",
             render_soul_card(tok)]
    if message:
        parts.append(f"<div class='card {html.escape(cls)}'>{html.escape(message)}</div>")
    parts.append("<div class=card><h3>Autonomy</h3>"
                 f"<p class=sub>Startup ceiling: level {int(doc.get('maximumLevel', 0))}. "
                 "A landing-page override can lower or raise the request only within "
                 "that ceiling; changing it does not restart vLLM.</p>"
                 f"<form method=post action='/soul/config?token={tok_q}'>"
                 f"<input type=hidden name=token value='{tok_q}'>"
                 "<label>Level <select name=autonomyLevel>")
    labels = {
        0: "0 — Off", 1: "1 — Observe", 2: "2 — Investigate",
        3: "3 — Verify",
    }
    for level, label in labels.items():
        selected = " selected" if level == int(config.get("requestedLevel", 0)) else ""
        disabled = " disabled" if level > int(doc.get("maximumLevel", 0)) else ""
        parts.append(f"<option value={level}{selected}{disabled}>{html.escape(label)}</option>")
    parts.append(
        "</select></label>"
        "<div class=sub style='margin-top:.5rem'>"
        "<b>1 Observe:</b> snapshots and journal from supplied evidence; no shell. "
        "<b>2 Investigate:</b> adds bounded read-only shell checks. "
        "<b>3 Verify:</b> adds idle-only inference canaries and a conditional "
        "long-context probe; it still cannot repair, restart, or change configuration."
        "</div>"
        f"<label> Snapshot interval (seconds) <input name=heartbeatIntervalS type=number "
        f"min=30 max=86400 value='{int(config.get('heartbeatIntervalS', 300))}'></label>"
        f"<label> Journal interval (seconds) <input name=journalIntervalS type=number "
        f"min=300 max=604800 value='{int(config.get('journalIntervalS', 3600))}'></label>"
        f"<label> Timezone <input name=timezone maxlength=64 "
        f"value='{html.escape(str(config.get('timezone', 'UTC')), quote=True)}'></label>"
        "<p><button type=submit>Save SOUL settings</button></p></form></div>")
    parts.append("<h2>Journal</h2>")
    if not entries:
        parts.append("<div class=card><p class=sub>No journal entries yet.</p></div>")
    for entry in entries:
        severity = html.escape(str(entry.get("severity", "info")))
        headline = html.escape(str(entry.get("headline", "Untitled")))
        stamp = html.escape(str(entry.get("timestamp", "")))
        summary = html.escape(str(entry.get("summary", "")))
        parts.append(
            f"<article class=card><h3>{headline}</h3><div class=sub>{stamp} "
            f"&middot; {severity} &middot; {html.escape(str(entry.get('kind', 'status')))}</div>"
            f"<p>{summary}</p>")
        observations = entry.get("observations") or []
        if observations:
            parts.append("<h3>Observations</h3><ul>" + "".join(
                "<li>" + html.escape(str(item)) + "</li>" for item in observations) + "</ul>")
        suggestions = entry.get("suggestions") or []
        if suggestions:
            parts.append("<h3>Suggestions</h3><ul>" + "".join(
                "<li>" + html.escape(str(item)) + "</li>" for item in suggestions) + "</ul>")
        for ref in entry.get("evidenceRefs") or []:
            evidence = _soul_evidence(ref)
            if evidence is None:
                continue
            encoded = json.dumps(evidence, ensure_ascii=False, indent=1)[:80_000]
            parts.append("<details><summary>Evidence "
                         + html.escape(str(ref)) + "</summary><pre>"
                         + html.escape(encoded) + "</pre></details>")
        parts.append("</article>")
    if entries:
        before = html.escape(str(entries[-1].get("id", "")), quote=True)
        parts.append(f"<p><a href='/soul?token={tok_q}&before={before}'>Older entries</a></p>")
    parts.append("</div></body></html>")
    return "".join(parts).encode()


def js_literal(value) -> str:
    """JSON string safe to embed inside an inline HTML script element."""
    return (json.dumps(value)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


def model_dir() -> str:
    """The entrypoint publishes the live model dir in the status file; it can
    move under the page's feet when the model-variant knob changes."""
    return status().get("model_dir") or MODEL_DIR


def weights_state() -> str:
    """Progress against the SELECTED model's size. This used to compare against a
    hardcoded 309 GiB — the GLM checkpoint — which made a 56 GiB Qwen download
    look like a GLM one and cost a live debugging session."""
    MODEL_DIR = model_dir()
    if os.path.isfile(os.path.join(MODEL_DIR, ".download-complete")):
        return "ready"
    done = 0
    for root, _dirs, files in os.walk(MODEL_DIR):
        for f in files:
            try:
                done += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    total = ""
    if gc is not None:
        try:
            eff, _s, _n = gc.resolve()
            variant = gc.VARIANTS.get(eff.get("MODEL_VARIANT"), {})
            if variant.get("download_gib"):
                total = f" of ~{variant['download_gib']} GiB"
        except Exception:
            pass
    return f"downloading — {done / 2**30:.0f}{total} GiB".replace(" GiB GiB", " GiB")


def engine_state() -> str:
    # vLLM may serve TLS or plain — probe both; any failure moves on
    for scheme, kw in (("https", {"context": ssl._create_unverified_context()}),
                       ("http", {})):
        try:
            with urllib.request.urlopen(f"{scheme}://localhost:{API_PORT}/health",
                                        timeout=1.5, **kw):
                return "serving"
        except Exception:
            continue
    phase = status().get("phase", "booting")
    return "starting (engine load + JIT warmup)" if phase == "starting-engine" else phase


SNIPPETS = [
    ("Oh My Pi (OMP 17+)", "~/.omp/agent/models.yml", """providers:
  turnkey:
    baseUrl: "$ep/v1"
    api: openai-completions
    apiKey: "$key"
    models:
      - id: "$model"
        name: "$model"
        reasoning: true
        input: $input
        supportsTools: true
        contextWindow: $ctx
        maxTokens: $output"""),
    ("opencode", "opencode.json (project or ~/.config/opencode/)", """{
  "$$schema": "https://opencode.ai/config.json",
  "provider": {
    "turnkey": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "$model",
      "options": {"baseURL": "$ep/v1", "apiKey": "$key"},
      "models": {
        "$model": {"name": "$model", "limit": {"context": $ctx, "output": $output}}
      }
    }
  }
}"""),
    ("Claude Code", "shell (native /v1/messages — Anthropic wire format)", """export ANTHROPIC_BASE_URL="$ep"
export ANTHROPIC_AUTH_TOKEN="$key"
export ANTHROPIC_MODEL="$model"
claude"""),
    ("Codex", "~/.codex/config.toml", """model = "$model"
model_provider = "turnkey"

[model_providers.turnkey]
name = "$model"
base_url = "$ep/v1"
env_key = "TURNKEY_API_KEY"
wire_api = "chat"

# then:  export TURNKEY_API_KEY="$key" """),
]

# Shared look. Literal CSS only (no stray $): safe inside plain strings.
STYLE = """<style>
:root{--bg:#f5f6fa;--card:#ffffff;--fg:#1c1e26;--muted:#6b7280;--line:#e5e7eb;
--accent:#4f7cff;--accent2:#9a6cff;--ok:#16a34a;--busy:#d97706;
--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media(prefers-color-scheme:dark){:root{--bg:#0e1013;--card:#171a20;--fg:#e7e9ee;
--muted:#8b93a3;--line:#262b35;--accent:#6d95ff;--accent2:#ae8bff;--ok:#34d17b;--busy:#f0a24b}}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--fg);
margin:0;line-height:1.5}
.wrap{max-width:62rem;margin:0 auto;padding:1.2rem 1rem 4rem}
header.hero{display:flex;align-items:baseline;gap:.8rem;flex-wrap:wrap;
padding:1.4rem 0 .6rem}
header.hero h1{margin:0;font-size:1.55rem;letter-spacing:-.02em}
header.hero h1 b{background:linear-gradient(90deg,var(--accent),var(--accent2));
-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--muted);font-size:.92rem}
.grid{display:grid;gap:.8rem;grid-template-columns:repeat(auto-fit,minmax(13rem,1fr));margin:.8rem 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:.9rem 1rem;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.card h3{margin:0 0 .3rem;font-size:.78rem;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted);font-weight:600}
.card .v{font-size:1.02rem;font-weight:600}
.ok{color:var(--ok)}.busy{color:var(--busy)}
.pill{display:inline-block;width:.55rem;height:.55rem;border-radius:50%;margin-right:.45rem}
.pill.ok{background:var(--ok)}.pill.busy{background:var(--busy);animation:pulse 1.6s infinite}
@keyframes pulse{50%{opacity:.35}}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
code{font-family:var(--mono);font-size:.88em;background:var(--bg);
border:1px solid var(--line);padding:.08rem .35rem;border-radius:6px}
pre{font-family:var(--mono);font-size:.82rem;background:var(--bg);
border:1px solid var(--line);padding:.8rem;border-radius:10px;overflow-x:auto;margin:.5rem 0}
h2{font-size:1.05rem;margin:1.6rem 0 .5rem}
details{background:var(--card);border:1px solid var(--line);border-radius:12px;
margin:.5rem 0;padding:.55rem .9rem}
summary{cursor:pointer;font-weight:600;font-size:.95rem}
button{font:inherit;font-size:.82rem;background:var(--card);color:var(--fg);
border:1px solid var(--line);border-radius:8px;padding:.3rem .8rem;cursor:pointer}
button:hover{border-color:var(--accent);color:var(--accent)}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.primary:hover{filter:brightness(1.08);color:#fff}
canvas.chart{width:100%;height:96px;display:block}
.layout{display:grid;gap:1rem;grid-template-columns:minmax(0,1fr)}
aside.chatpane{display:none}
.wrap.wide{max-width:64rem}
@media(min-width:75rem){
 .wrap.wide{max-width:96rem}
 .wrap.wide .layout{grid-template-columns:minmax(0,1fr) 26rem}
 .wrap.wide aside.chatpane{display:flex;flex-direction:column;position:sticky;top:1rem;
  height:calc(100vh - 2rem);background:var(--card);border:1px solid var(--line);
  border-radius:14px;padding:.9rem;overflow:hidden}
}
aside.chatpane h2{margin:0 0 .5rem;font-size:.95rem}
.chatpane #log{flex:1;overflow-y:auto;min-height:0}
.msg{max-width:85%;padding:.6rem .9rem;border-radius:14px;margin:.35rem 0;
white-space:pre-wrap;font-size:.92rem;width:fit-content}
.msg.you{background:linear-gradient(120deg,var(--accent),var(--accent2));color:#fff;
margin-left:auto;border-bottom-right-radius:4px}
.msg.bot{background:var(--bg);border:1px solid var(--line);border-bottom-left-radius:4px}
details.think{color:var(--muted);font-size:.8em;margin:.2rem 0;max-width:85%;
background:none;border:none;padding:0 .3rem}
.composer{display:flex;gap:.5rem;align-items:flex-end;padding-top:.5rem;
border-top:1px solid var(--line)}
#in{flex:1;font:inherit;background:var(--bg);color:var(--fg);
border:1px solid var(--line);border-radius:12px;padding:.5rem .7rem;
min-height:2.6rem;max-height:9rem;resize:vertical}
.side{display:flex;flex-direction:column;gap:.35rem}
label.thinkbox{font-size:.78rem;color:var(--muted);white-space:nowrap}
.chartv{font-family:var(--mono);font-size:1.02rem;font-weight:700;float:right}
.legend{font-size:.72rem;color:var(--muted)}
.legend i{display:inline-block;width:.9em;height:3px;border-radius:2px;
vertical-align:middle;margin:0 .3em 0 .8em}
.chatempty{height:100%;min-height:10rem;display:grid;place-content:center;text-align:center;
padding:1.25rem;color:var(--muted)}
.chatempty-icon{font-size:1.4rem;color:var(--accent);margin-bottom:.45rem}
.chatempty strong{display:block;color:var(--fg);font-size:.9rem;margin-bottom:.2rem}
.chatempty span{display:block;font-size:.76rem;max-width:15rem}
.composer{display:block;padding:.65rem;border-top:1px solid var(--line)}
.inputrow{display:flex;gap:.5rem;align-items:flex-end}
.composerbar{display:flex;align-items:center;justify-content:space-between;gap:.6rem;margin-top:.5rem}
.chatprefs,.composer-actions{display:flex;align-items:center;gap:.55rem;flex-wrap:wrap}
.turncount{color:var(--muted);font-size:.7rem}
.thinkbox{position:relative;display:flex;align-items:center;gap:.35rem;color:var(--muted);
font-size:.73rem;white-space:nowrap;cursor:pointer;user-select:none}
.thinkbox input{position:absolute;opacity:0;pointer-events:none}
.switch{position:relative;width:1.8rem;height:1rem;border-radius:99px;background:var(--line)}
.switch:after{content:"";position:absolute;top:.13rem;left:.13rem;width:.74rem;height:.74rem;
border-radius:50%;background:var(--card);box-shadow:0 1px 3px rgba(0,0,0,.2);
transition:transform .18s ease}
.thinkbox input:checked+.switch{background:var(--accent)}
.thinkbox input:checked+.switch:after{transform:translateX(.8rem)}
@media(max-width:38rem){.composerbar{align-items:flex-start;flex-direction:column}
.composer-actions{width:100%;justify-content:flex-end}}
</style>"""

# Live dashboard: the browser scrapes $ep/metrics (Bearer $key unless placeholder).
METRICS_SECTION = Template("""
<h2>Live performance</h2>
<div class=grid>
 <div class=card><h3>Throughput <span class=chartv id=v0></span></h3>
   <canvas class=chart id=c0></canvas>
   <div class=legend><i style="background:var(--accent)"></i>generate tok/s
   <i style="background:var(--accent2)"></i>prompt tok/s</div></div>
 <div class=card><h3>Requests <span class=chartv id=v1></span></h3>
   <canvas class=chart id=c1></canvas>
   <div class=legend><i style="background:var(--accent)"></i>running
   <i style="background:var(--busy)"></i>waiting</div></div>
 <div class=card><h3>KV cache used <span class=chartv id=v2></span></h3>
   <canvas class=chart id=c2></canvas></div>
 <div class=card><h3>Prefix-cache hit rate <span class=chartv id=v3></span></h3>
   <canvas class=chart id=c3></canvas></div>
</div>
<script>
(function(){
var EP=$ep_js, KEY=$key_js;
var HDRS = KEY.charAt(0)==="<" ? {} : {"Authorization":"Bearer "+KEY};
var N=100, S={gen:[],pro:[],run:[],wai:[],kv:[],hit:[]};
var prev=null, prevT=0;
fetch(EP+"/v1/models",{headers:HDRS}).then(function(r){return r.json()}).then(function(j){
  document.getElementById("modeljson").textContent=JSON.stringify(j,null,2);
}).catch(function(){document.getElementById("modeljson").textContent="failed to fetch /v1/models"});
function parse(text){
  var v={}, lines=text.split("\\n");
  for(var i=0;i<lines.length;i++){
    var ln=lines[i];
    if(!ln||ln.charCodeAt(0)===35)continue;
    var sp=ln.lastIndexOf(" ");
    var name=ln.slice(0,sp), val=parseFloat(ln.slice(sp+1));
    var br=name.indexOf("{"); if(br>=0)name=name.slice(0,br);
    if(!isNaN(val))v[name]=(v[name]||0)+val;
  }
  return v;
}
function push(a,x){a.push(x);if(a.length>N)a.shift()}
function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
function draw(id,series,colors){
  var c=document.getElementById(id),dpr=window.devicePixelRatio||1;
  var w=c.clientWidth,h=c.clientHeight;
  c.width=w*dpr;c.height=h*dpr;
  var g=c.getContext("2d");g.scale(dpr,dpr);g.clearRect(0,0,w,h);
  var mx=1e-9,si,i;
  for(si=0;si<series.length;si++)for(i=0;i<series[si].length;i++)
    if(series[si][i]>mx)mx=series[si][i];
  mx*=1.15;
  g.strokeStyle="rgba(128,128,128,.25)";g.lineWidth=1;
  g.beginPath();g.moveTo(0,h-.5);g.lineTo(w,h-.5);g.stroke();
  for(si=series.length-1;si>=0;si--){
    var s=series[si]; if(s.length<2)continue;
    g.strokeStyle=colors[si];g.lineWidth=2;g.lineJoin="round";g.beginPath();
    for(i=0;i<s.length;i++){
      var px=i/(N-1)*w, py=h-4-(s[i]/mx)*(h-10);
      i?g.lineTo(px,py):g.moveTo(px,py);
    }
    g.stroke();
    if(si===0){
      var grad=g.createLinearGradient(0,0,0,h);
      grad.addColorStop(0,colors[0]+"55");grad.addColorStop(1,colors[0]+"00");
      g.lineTo((s.length-1)/(N-1)*w,h);g.lineTo(0,h);g.closePath();
      g.fillStyle=grad;g.fill();
    }
  }
}
function fmt(x){return x>=1000?(x/1000).toFixed(1)+"k":x.toFixed(0)}
function last(a){return a.length?a[a.length-1]:0}
function tick(){
  fetch(EP+"/metrics",{headers:HDRS}).then(function(r){
    if(!r.ok)throw 0; return r.text();
  }).then(function(text){
    var v=parse(text), t=Date.now()/1000;
    if(prev){
      var dt=Math.max(t-prevT,.1);
      push(S.gen,Math.max(0,(v["vllm:generation_tokens_total"]-prev["vllm:generation_tokens_total"])/dt||0));
      push(S.pro,Math.max(0,(v["vllm:prompt_tokens_total"]-prev["vllm:prompt_tokens_total"])/dt||0));
      var dq=v["vllm:prefix_cache_queries_total"]-prev["vllm:prefix_cache_queries_total"];
      var dh=v["vllm:prefix_cache_hits_total"]-prev["vllm:prefix_cache_hits_total"];
      push(S.hit,dq>0?100*dh/dq:last(S.hit));
    }
    push(S.run,v["vllm:num_requests_running"]||0);
    push(S.wai,v["vllm:num_requests_waiting"]||0);
    push(S.kv,100*(v["vllm:kv_cache_usage_perc"]||0));
    prev=v;prevT=t;
    var A=css("--accent"),A2=css("--accent2"),B=css("--busy");
    draw("c0",[S.gen,S.pro],[A,A2]);
    draw("c1",[S.run,S.wai],[A,B]);
    draw("c2",[S.kv],[A]);
    draw("c3",[S.hit],[A2]);
    document.getElementById("v0").textContent=S.gen.length?fmt(last(S.gen))+" tok/s":"";
    document.getElementById("v1").textContent=last(S.run)+" / "+last(S.wai);
    document.getElementById("v2").textContent=S.kv.length?last(S.kv).toFixed(1)+"%":"";
    document.getElementById("v3").textContent=S.hit.length?last(S.hit).toFixed(0)+"%":"";
  }).catch(function(){}).then(function(){setTimeout(tick,3000)});
}
tick();
})();
</script>""")


CHAT_WIDGET = """<div id=log aria-live=polite>
 <div class=chatempty id=chatempty>
  <div><div class=chatempty-icon aria-hidden=true>✦</div>
  <strong>Start a conversation</strong>
  <span>Follow up naturally — this chat keeps the conversation until you clear it.</span></div>
 </div>
</div>
<div class=composer>
 <div class=inputrow>
  <textarea id=in rows=2 aria-label="Chat message"
   placeholder="Message the model…"></textarea>
  <button id=send class=primary type=button>Send</button>
 </div>
 <div class=composerbar>
  <div class=chatprefs>
   <label class=thinkbox title="Ask the model to reason before answering">
    <input type=checkbox id=think checked><span class=switch aria-hidden=true></span>
    Thinking
   </label>
   <label class=thinkbox title="Include saved reasoning in later turns">
    <input type=checkbox id=preserve><span class=switch aria-hidden=true></span>
    Preserve thinking
   </label>
  </div>
  <div class=composer-actions>
   <span class=turncount id=turns>New conversation</span>
   <button id=stop type=button disabled>Stop</button>
   <button id=clear type=button>Clear</button>
  </div>
 </div>
</div>"""

CHAT_JS = """<script>
const EP=$ep_js, KEY=$key_js, MODEL=$model_js, msgs=[];
const log=document.getElementById("log"), inp=document.getElementById("in");
let ctrl=null;
function el(tag,cls,txt){
  const empty=document.getElementById("chatempty");if(empty)empty.remove();
  const e=document.createElement(tag);if(cls)e.className=cls;
  if(txt)e.textContent=txt;log.appendChild(e);log.scrollTop=log.scrollHeight;return e
}
function emptyState(){
  log.innerHTML='<div class="chatempty" id="chatempty"><div><div class="chatempty-icon" aria-hidden="true">✦</div><strong>Start a conversation</strong><span>Follow up naturally — this chat keeps the conversation until you clear it.</span></div></div>'
}
function wireMessages(){
  const keep=document.getElementById("preserve").checked;
  return msgs.map(m=>{
    const wire={role:m.role,content:m.content};
    if(keep&&m.role==="assistant"&&m.reasoning&&!m.reasoningOnly)
      wire[m.reasoningField||"reasoning_content"]=m.reasoning;
    return wire;
  });
}
function updateTurns(){
  const n=msgs.filter(m=>m.role==="assistant").length;
  document.getElementById("turns").textContent=n?n+" turn"+(n===1?"":"s"):"New conversation";
}
async function send(){
  const text=inp.value.trim(); if(!text||ctrl)return;
  inp.value="";inp.style.height="";msgs.push({role:"user",content:text});el("div","msg you",text);
  const think=el("details","think"); think.appendChild(document.createElement("summary")).textContent="thinking…";
  const tbody=think.appendChild(document.createElement("div"));
  const out=el("div","msg bot","");
  ctrl=new AbortController();document.getElementById("stop").disabled=false;
  document.getElementById("send").disabled=true;
  let answer="",reasoning="",reasoningField="",failed=false;
  try{
    const r=await fetch(EP+"/v1/chat/completions",{method:"POST",signal:ctrl.signal,
      headers:{"Authorization":"Bearer "+KEY,"Content-Type":"application/json"},
      body:JSON.stringify({model:MODEL,messages:wireMessages(),stream:true,max_tokens:$output,
        chat_template_kwargs:{enable_thinking:document.getElementById("think").checked}})});
    if(!r.ok){failed=true;out.textContent="HTTP "+r.status+": "+await r.text();return}
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf="";
    for(;;){const {done,value}=await rd.read(); if(done)break;
      buf+=dec.decode(value,{stream:true}); const lines=buf.split("\\n"); buf=lines.pop();
      for(const ln of lines){ if(!ln.startsWith("data: ")||ln.includes("[DONE]"))continue;
        let packet;try{packet=JSON.parse(ln.slice(6))}catch(_){continue}
        const ch=packet.choices;
        if(!ch||!ch.length)continue; // final usage chunk has choices:[]
        const d=ch[0].delta||{};
        const thought=d.reasoning_content??d.reasoning;
        if(thought){
          reasoningField=d.reasoning_content!=null?"reasoning_content":"reasoning";
          reasoning+=thought;tbody.textContent=reasoning;think.open=true
        }
        if(d.content){answer+=d.content;out.textContent=answer;
          if(think.open){think.open=false;think.querySelector("summary").textContent="thinking ("+reasoning.length+" chars)"}}
        log.scrollTop=log.scrollHeight;}}
  }catch(e){if(e.name!=="AbortError"){failed=true;out.textContent="Request failed. "+e}}
  finally{
    const stopped=ctrl&&ctrl.signal.aborted,finalText=answer||reasoning;
    ctrl=null;document.getElementById("stop").disabled=true;
    document.getElementById("send").disabled=false;
    if(!answer&&reasoning){out.textContent=reasoning;think.remove()}
    else if(!reasoning)think.remove();
    else think.querySelector("summary").textContent="thinking ("+reasoning.length+" chars)";
    if(!finalText&&!failed)out.textContent=stopped?"Generation stopped.":"The model returned no text.";
    if(finalText){
      const preserve=document.getElementById("preserve").checked;
      msgs.push({role:"assistant",content:finalText,
        reasoning:preserve&&answer?reasoning:"",
        reasoningField:reasoningField||"reasoning_content",
        reasoningOnly:!answer&&!!reasoning});
      updateTurns();
    }
    else if(msgs.length&&msgs[msgs.length-1].role==="user")msgs.pop();
    inp.focus();
  }
}
document.getElementById("send").onclick=send;
document.getElementById("stop").onclick=()=>ctrl&&ctrl.abort();
document.getElementById("clear").onclick=()=>{msgs.length=0;emptyState();updateTurns();inp.focus()};
inp.addEventListener("keydown",e=>{if(e.ctrlKey&&e.key==="Enter")send()});
inp.addEventListener("input",()=>{inp.style.height="";inp.style.height=Math.min(inp.scrollHeight,144)+"px"});
</script>"""

CHAT_PAGE = Template("""<!doctype html><html><head><title>$model_html chat</title>
<meta name=viewport content="width=device-width,initial-scale=1">""" + STYLE + """<style>
.wrap{max-width:52rem;display:flex;flex-direction:column;height:100vh;padding-bottom:1rem}
#log{flex:1;overflow-y:auto;padding:.5rem 0}
.msg.bot{background:var(--card)}
</style></head><body><div class=wrap>
<header class=hero><h1><b>$model_html</b> quick chat</h1>
<span class=sub><a href="/?token=$token_html">&larr; status &amp; dashboard</a></span></header>
""" + CHAT_WIDGET + CHAT_JS + """<script>inp.focus();</script></div></body></html>""")

def page_head(title="vLLM turnkey"):
    return ("<!doctype html><html><head><title>" + html.escape(title) + "</title>"
            '<meta name=viewport content="width=device-width,initial-scale=1">'
            + STYLE + "</head><body>")

# ==========================================================================
# self-service configuration
# ==========================================================================
# Editing what the engine serves is at least as privileged as reading the API
# key, so the editor is gated on the same OPEN_BUTTON_TOKEN as /chat. It does
# NOT additionally require TLS: on a rental without a DNS provider there is no
# cert, and refusing to work there would mean the feature exists only for the
# minority who set up deSEC. The page says so instead of pretending.

CONFIG_STYLE = """<style>
table.cfg{width:100%;border-collapse:collapse;font-size:.9rem}
table.cfg td{padding:.45rem .5rem;border-top:1px solid var(--line);vertical-align:top}
table.cfg td.k{white-space:nowrap;font-family:var(--mono);font-size:.82rem}
table.cfg td.v{width:14rem}
table.cfg input,table.cfg select{font:inherit;font-size:.85rem;width:100%;
background:var(--bg);color:var(--fg);border:1px solid var(--line);
border-radius:8px;padding:.25rem .4rem}
table.cfg input:disabled,table.cfg select:disabled{opacity:.6}
.src{font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;
border:1px solid var(--line);border-radius:99px;padding:.05rem .45rem;color:var(--muted)}
.src.file{border-color:var(--accent);color:var(--accent)}
.src.env{border-color:var(--accent2);color:var(--accent2)}
.why{color:var(--muted);font-size:.82rem;margin-top:.25rem}
.finding{border-left:3px solid var(--busy);padding:.4rem .7rem;margin:.4rem 0;
background:var(--card);border-radius:0 8px 8px 0;font-size:.87rem}
.finding.error{border-left-color:#e5484d}
.banner{padding:.7rem .9rem;border-radius:10px;margin:.6rem 0;font-size:.9rem;
background:var(--card);border:1px solid var(--line)}
.banner.bad{border-color:#e5484d}
.banner.good{border-color:var(--ok)}
textarea.imp{width:100%;min-height:7rem;font-family:var(--mono);font-size:.8rem;
background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:10px;padding:.5rem}
</style>"""


def _val_ctx():
    """What validate() needs from the host: GPU count and what the state file
    actually persists."""
    ctx = {"vision_available": os.path.exists(
        os.path.join(model_dir(), "vision_tower.safetensors"))}
    try:
        ctx["gpu_count"] = int(os.environ["GLM_GPU_COUNT"])
    except (KeyError, ValueError):
        pass
    try:
        ctx["state_keys"] = list(gc.load_state_file())
    except Exception:
        ctx["state_keys"] = []
    return ctx


def boot_notes(limit=40):
    """The entrypoint's own warnings. RunPod has no console view and no CLI log
    command, so a boot warning that only reaches stdout is invisible exactly
    when it matters most."""
    if gc is None:
        return []
    try:
        with open(os.path.join(gc.runtime_dir(), "boot-notes.log")) as f:
            return [ln.rstrip() for ln in f][-limit:]
    except OSError:
        return []


def deployment() -> dict:
    """What is ACTUALLY configured, for every user-visible string.

    The header used to advertise a fixed model name, context length, KV dtype,
    speculation depth and GPU model regardless of what was actually running.
    Under another family or another GPU count every clause of that is false, and
    on a live pod it caused two correctly-honoured knobs to be reported as
    ignored."""
    st = status()
    out = {"title": "vLLM turnkey", "model": "", "family": "", "chips": [],
           "gpus": st.get("gpus", ""), "gpu_name": st.get("gpu_name", ""),
           "provider": st.get("provider", "unknown"), "sshd": st.get("sshd", "")}
    if gc is None:
        return out
    try:
        eff, _src, _notes = gc.resolve()
    except Exception:
        return out
    fam = gc.family(eff.get("MODEL_FAMILY"))
    variant = gc.VARIANTS.get(eff.get("MODEL_VARIANT"), {})
    served = str(eff.get("SERVED_MODEL_NAME", "")).split()[0] or "model"
    out["model"] = served
    out["family"] = fam["label"]
    out["title"] = f"{served} turnkey"
    out["context"] = int(eff.get("MAX_MODEL_LEN") or 0)
    out["output"] = int(eff.get("MODEL_OUTPUT_LIMIT") or 8192)
    is_vision = (bool(eff.get("MULTIMODAL"))
                 if eff.get("MODEL_FAMILY") == "qwen36"
                 else bool(eff.get("VISION")))
    out["input"] = "[text, image]" if is_vision else "[text]"
    out["served_names"] = str(eff.get("SERVED_MODEL_NAME", served)).split()
    ctx = out["context"]
    ctx_s = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
    chips = [f"{ctx_s} context", f"{eff.get('KV_CACHE_DTYPE')} KV"]
    if eff.get("MTP_TOKENS"):
        chips.append(f"MTP-{eff['MTP_TOKENS']}")
    else:
        chips.append("no speculation")
    tp = eff.get("TENSOR_PARALLEL_SIZE")
    gpus = out["gpus"] or tp
    chips.append(f"TP={tp}" + (f" of {gpus} GPU" + ("s" if str(gpus) != "1" else "")
                               if str(gpus) != str(tp) else
                               f" x {out['gpu_name'] or 'GPU'}"))
    out["chips"] = chips
    out["repo"] = gc.derive(eff).get("MODEL_REPO", variant.get("repo", ""))
    out["tested"] = fam.get("tested", True)
    return out


def config_enabled() -> bool:
    return gc is not None and bool(TOKEN)


def verify_last() -> dict:
    return gc.read_json(gc.p_verify_last()) if gc else {}


def correctness() -> tuple:
    """-> (headline, ok_bool, detail). Never collapses 'answered a short prompt'
    into 'healthy': the failure modes this appliance has actually seen all pass
    short prompts and fail past ~32K tokens."""
    v = verify_last()
    if not v:
        return ("not yet verified", False,
                "the post-start probe has not reported yet")
    if not v.get("health"):
        return ("engine did not come up", False, v.get("reason", ""))
    if not (v.get("short_prompt") or {}).get("ok"):
        return ("short-prompt checks FAILED", False, v.get("reason", ""))
    lc = v.get("long_context") or {}
    if not lc.get("attempted"):
        return ("short prompts only — long context UNVERIFIED", False,
                lc.get("detail", "the long-context probe did not run"))
    if lc.get("ok"):
        return ("verified incl. long context (%s/%s codes at ~%s tok)"
                % (lc.get("found"), lc.get("total"), lc.get("tokens")), True,
                v.get("reason", ""))
    return ("LONG-CONTEXT PROBE FAILED (%s/%s codes)"
            % (lc.get("found", 0), lc.get("total", 0)), False, v.get("reason", ""))


def _field(knob, value, source=""):
    key = html.escape(knob["key"])
    editable = knob.get("editable", True) and source != "n/a"
    dis = "" if editable else " disabled"
    if source == "n/a":
        return ("<span class=sub>not applicable to this model family</span>"
                f"<input type=hidden name='__na_{key}' value=1>")
    if knob["type"] == "bool":
        opts = "".join(
            "<option value='%s'%s>%s</option>" % (
                v, " selected" if bool(value) == (v == "1") else "", lbl)
            for v, lbl in (("1", "on"), ("0", "off")))
        return f"<select name='{key}'{dis}>{opts}</select>"
    if knob["type"] == "choice":
        labels = {}
        if knob["key"] == "MODEL_FAMILY":
            labels = {k: v["label"] for k, v in gc.FAMILIES.items()}
        elif knob["key"] == "MODEL_VARIANT":
            labels = {k: "%s [%s]" % (v["label"], v.get("family", "?"))
                      for k, v in gc.VARIANTS.items()}
        elif knob["key"] == "MTP_DRAFT":
            labels = {k: v["label"] for k, v in gc.DRAFTS.items()}
        opts = "".join(
            "<option value='%s'%s>%s</option>" % (
                html.escape(c), " selected" if str(value) == c else "",
                html.escape(labels.get(c, c)))
            for c in knob["choices"])
        return f"<select name='{key}'{dis}>{opts}</select>"
    if knob["type"] in ("int", "float"):
        step = "1" if knob["type"] == "int" else "0.01"
        lo = f" min='{knob['min']}'" if "min" in knob else ""
        hi = f" max='{knob['max']}'" if "max" in knob else ""
        return (f"<input type=number step='{step}'{lo}{hi} name='{key}' "
                f"value='{html.escape(gc.to_text(knob, value))}'{dis}>")
    return (f"<input type=text name='{key}' "
            f"value='{html.escape(gc.to_text(knob, value))}'{dis}>")


def _findings_html(findings):
    out = []
    for f in findings:
        cls = "finding error" if f["level"] == "error" else "finding"
        keys = ", ".join(f["keys"])
        out.append(f"<div class='{cls}'><b>{html.escape(f['level'].upper())}</b> "
                   f"<code>{html.escape(keys)}</code><br>{html.escape(f['message'])}</div>")
    return "".join(out)


def _failures_html():
    root = gc.p_failures()
    try:
        dirs = sorted(os.listdir(root), reverse=True)[:5]
    except OSError:
        return ""
    out = []
    for d in dirs:
        fdir = os.path.join(root, d)
        meta = gc.read_json(os.path.join(fdir, "meta.json"))
        try:
            with open(os.path.join(fdir, "diff.txt")) as f:
                diff = f.read()
        except OSError:
            diff = ""
        analysis = ""
        try:
            with open(os.path.join(fdir, "analysis.md")) as f:
                analysis = f.read()
        except OSError:
            marker = os.path.join(gc.state_dir(), "soul", "incidents",
                                  "rollback-" + d + ".json")
            soul_analysis = gc.read_json(marker).get("record", {})
            if soul_analysis:
                analysis = (str(soul_analysis.get("headline", "")) + "\n\n"
                            + str(soul_analysis.get("summary", "")))
                suggestions = soul_analysis.get("suggestions") or []
                if suggestions:
                    analysis += "\n\nSuggestions:\n" + "\n".join(
                        "- " + str(item) for item in suggestions)
            else:
                analysis = ("_The running model has not produced an explanation yet "
                            "(it is generated once the known-good config is serving)._")
        sigs = "".join(f"<li>{html.escape(s)}</li>" for s in meta.get("signatures") or [])
        out.append(
            "<details><summary>%s &mdash; %s</summary>"
            "<p class=sub>%s</p>%s"
            "<h3 style='font-size:.9rem;margin:.6rem 0 .2rem'>What the model says</h3>"
            "<pre>%s</pre>"
            "<h3 style='font-size:.9rem;margin:.6rem 0 .2rem'>Config diff</h3><pre>%s</pre>"
            "<p class=sub>Full log: <code>%s/error.log</code></p></details>" % (
                html.escape(d), html.escape(meta.get("reason", "?")),
                html.escape("known failure signatures matched:") if sigs else "",
                f"<ul>{sigs}</ul>" if sigs else "",
                html.escape(analysis), html.escape(diff or "(none)"),
                html.escape(fdir)))
    return "".join(out)


def render_config(tok: str, secure: bool, banner=None, banner_cls="",
                  extra_findings=None) -> bytes:
    effective, sources, notes = gc.resolve()
    findings = list(extra_findings or []) + gc.validate(effective, _val_ctx())
    fam = gc.family(effective.get("MODEL_FAMILY"))
    st = gc.apply_state()
    head, ok, detail = correctness()
    tok_q = html.escape(tok, quote=True)

    dep = deployment()
    parts = ["<!doctype html><html><head><title>"
             + html.escape(dep["model"] or "vLLM") + " configuration</title>"
             "<meta name=viewport content='width=device-width,initial-scale=1'>",
             STYLE, CONFIG_STYLE, "</head><body><div class=wrap>",
             "<header class=hero><h1><b>"
             + html.escape(dep["model"] or "vLLM") + "</b> configuration</h1>"
             f"<span class=sub><a href='/?token={tok_q}'>&larr; status &amp; dashboard</a>"
             + (f" &middot; <a href='/terminate?token={tok_q}'>terminate</a>"
                if terminate_available() else "")
             + "</span></header>"]
    if banner:
        parts.append(f"<div class='banner {banner_cls}'>{banner}</div>")
    if not secure:
        parts.append("<div class=banner>This page is being served over plain HTTP. "
                     "The token and everything you type here cross the network in the "
                     "clear. Set <code>DESEC_TOKEN</code>+<code>DESEC_DOMAIN</code> "
                     "(or <code>ACME_DOMAIN</code>) to get TLS.</div>")

    cls = "ok" if ok else "busy"
    parts.append(
        "<div class=grid>"
        "<div class=card><h3>Apply state</h3><div class=v>%s</div>"
        "<div class=why>%s</div></div>"
        "<div class=card><h3>Correctness</h3>"
        "<div class=v><span class='pill %s'></span><span class='%s'>%s</span></div>"
        "<div class=why>%s</div></div></div>" % (
            html.escape(st.get("mode", "steady")),
            html.escape(str(st.get("detail") or "")),
            cls, cls, html.escape(head), html.escape(detail)))

    if st.get("mode") == "rolled-back":
        parts.append(
            "<div class='banner bad'><b>The last configuration you applied was rolled "
            "back.</b> The previous known-good configuration is what is running now. "
            "The failed configuration, its boot log and a diff are kept below.</div>")

    parts.append(
        "<div class=card><h3>Model family</h3><div class=v>%s</div>"
        "<div class=why>%s</div>%s</div>" % (
            html.escape(fam["label"]), html.escape(fam["notes"]),
            "" if fam.get("tested") else
            "<div class=finding error><b>UNTESTED</b> — nobody has booted this image "
            "with this family. Treat every default as a starting point.</div>"))
    if notes:
        parts.append("".join(f"<div class=finding>{html.escape(n)}</div>" for n in notes))
    if findings:
        parts.append("<h2>Pre-validation</h2>" + _findings_html(findings))

    parts.append(f"<form method=post action='/config/apply?token={tok_q}'>"
                 f"<input type=hidden name=token value='{tok_q}'>")
    group = None
    parts.append("<table class=cfg>")
    for knob in gc.KNOBS:
        if knob.get("group") != group:
            group = knob.get("group")
            parts.append(f"<tr><td colspan=3><h2>{html.escape(group)}</h2></td></tr>")
        key = knob["key"]
        src = sources[key]
        locked = "" if knob.get("editable", True) else " <span class=src>locked</span>"
        na = " <span class=src>n/a</span>" if src == "n/a" else ""
        parts.append(
            "<tr><td class=k>%s%s%s</td><td class=v>%s</td><td>"
            "<b>%s</b> <span class='src %s'>%s</span><div class=why>%s</div></td></tr>" % (
                html.escape(key), locked, na, _field(knob, effective[key], src),
                html.escape(knob.get("label", key)), src, src,
                html.escape(knob["rationale"])))
    parts.append("</table>")
    parts.append(
        "<p><button class=primary type=submit>Apply &amp; restart vLLM</button> "
        f"<a href='/config/export?token={tok_q}'><button type=button>Export JSON"
        "</button></a></p>"
        "<p class=sub>Apply writes the JSON state file on the volume and restarts the "
        "engine only &mdash; the container, the weights and the API key are untouched. "
        "A restart costs a few minutes of engine load; changing the model variant, the "
        "draft type or vision can cost a download on top. If the new configuration does "
        "not come up, or comes up and fails the long-context probe, it is rolled back "
        "automatically.</p></form>")

    parts.append(
        f"<h2>Import</h2><form method=post action='/config/import?token={tok_q}'>"
        f"<input type=hidden name=token value='{tok_q}'>"
        "<textarea class=imp name=doc placeholder='Paste a previously exported "
        "config JSON here'></textarea>"
        "<p><button type=submit>Import, validate &amp; restart</button></p></form>")

    parts.append(
        f"<h2>Reset</h2><form method=post action='/config/reset?token={tok_q}'>"
        f"<input type=hidden name=token value='{tok_q}'>"
        "<p class=sub>Deletes the state file, returning to the template environment "
        "and the built-in defaults.</p>"
        "<p><button type=submit>Reset to defaults &amp; restart</button></p></form>")

    fails = _failures_html()
    if fails:
        parts.append("<h2>Rejected configurations</h2>" + fails)

    parts.append("</div></body></html>")
    return "".join(parts).encode()


def _form_values(form: dict):
    """Coerce a submitted form into knob values, collecting per-field errors."""
    values, errs = {}, []
    for knob in gc.KNOBS:
        if not knob.get("editable", True):
            continue
        raw = form.get(knob["key"], [None])[0]
        if raw is None:
            continue
        try:
            values[knob["key"]] = gc.coerce(knob, raw)
        except gc.ConfigError as e:
            errs.append({"id": "coerce", "level": "error", "keys": [knob["key"]],
                         "message": str(e)})
    return values, errs


def _coerce_imported_values(values: dict):
    """Coerce imported JSON knob values through gc.coerce, collecting per-field
    errors exactly like _form_values does for a submitted form — so a bad
    import surfaces the error instead of being silently dropped by minimize."""
    coerced, errs = {}, []
    knobs = {knob["key"]: knob for knob in gc.KNOBS}
    for key, raw in values.items():
        knob = knobs.get(key)
        if knob is None or not knob.get("editable", True):
            continue
        try:
            coerced[key] = gc.coerce(knob, raw)
        except gc.ConfigError as e:
            errs.append({"id": "coerce", "level": "error", "keys": [key],
                         "message": str(e)})
    return coerced, errs


def apply_values(values: dict):
    """Validate a candidate, persist it, and ask the supervisor for a restart.

    -> (ok, findings, message). Nothing is written when validation finds an
    error: the whole point is that the user never reaches a failed restart for
    a reason that was knowable beforehand."""
    try:
        gc.check_forbidden(values)
    except gc.ConfigError as e:
        return False, [{"id": "forbidden-key", "level": "error",
                        "keys": list(gc.FORBIDDEN_STATE_KEYS), "message": str(e)}], \
            "Nothing was applied."
    dropped = []
    minimal = gc.minimize(values, dropped)
    effective, _sources, notes = gc.resolve(state_values=minimal)
    ctx = _val_ctx()
    ctx["state_keys"] = list(minimal)
    findings = gc.validate(effective, ctx)
    if gc.errors(findings):
        return False, findings, "Nothing was applied — fix the errors below."
    # Report dropped knobs on EVERY path, including "no change" — otherwise a
    # user who submits a GLM knob while on Qwen is told nothing happened, which
    # is true but hides the more useful fact of WHY.
    drop_msg = ""
    if dropped:
        drop_msg = ("<p><b>Not applicable to this model family, so not saved:</b> "
                    + html.escape(", ".join(sorted(set(dropped)))) + "</p>")
    good = gc.read_json(gc.p_known_good())
    current, _s, _n = gc.resolve()
    if not gc.diff(current, effective):
        return True, findings, ("No change: the submitted configuration is the one "
                                "running." + drop_msg)
    gc.write_json_atomic(gc.p_state(), {
        "values": minimal,
        "written_at": gc.utcnow_iso()}, mode=0o600)
    gc.set_apply_state("trial", since=gc.utcnow_iso(),
                       detail="restarting into a candidate configuration",
                       baseline=good.get("ts"))
    os.makedirs(gc.runtime_dir(), exist_ok=True)
    with open(gc.p_restart_flag(), "w") as f:
        f.write("landing-page\n")
    msg = ("Applied. vLLM is restarting with:<br><pre>"
           + html.escape(gc.diff_text(current, effective)) + "</pre>")
    # Not silently ignored: the user is told which knobs the selected family
    # does not have and that they were not written.
    return True, findings, msg + drop_msg


# ==========================================================================
# terminate the instance
# ==========================================================================

TERMINATE_STYLE = """<style>
.danger{border:1px solid #e5484d;border-radius:12px;padding:1rem;margin:1rem 0;
background:var(--card)}
.danger h2{margin-top:0;color:#e5484d}
.danger ul{margin:.3rem 0 .6rem 1.1rem;padding:0}
.danger li{margin:.15rem 0}
button.destroy{background:#e5484d;border-color:#e5484d;color:#fff;font-weight:600}
button.destroy:hover{filter:brightness(1.1);color:#fff}
button.destroy:disabled{opacity:.45;cursor:not-allowed;filter:none}
.lockbox{border:1px solid var(--line);border-radius:12px;padding:.8rem 1rem;
margin:1rem 0;background:var(--card)}
input.confirm{font-family:var(--mono);font-size:.95rem;width:100%;max-width:22rem;
background:var(--bg);color:var(--fg);border:1px solid var(--line);
border-radius:8px;padding:.4rem .6rem}
.kv{font-size:.88rem}.kv b{font-family:var(--mono);font-weight:600}
</style>"""


def terminate_available() -> bool:
    """The control only exists when the shared modules imported AND the page is
    token-gated. Termination must never be reachable unauthenticated."""
    return bool(gc is not None and prov is not None and terminate_worker is not None
                and TOKEN)


def _switch_html():
    st = gc.read_switches()
    allowed, why = gc.termination_allowed()
    rows = [
        ("Kill switch (TERMINATE_ENABLED)",
         "on — the terminate control is available" if st["enabled"]
         else "off — terminate from this page is disabled (this is the default)"),
        ("Anti-kill switch (TERMINATE_LOCKED)",
         "LOCKED — termination is refused no matter what" if st["locked"]
         else "not locked"),
    ]
    body = "".join(f"<div class=kv><b>{html.escape(k)}</b>: {html.escape(v)}</div>"
                   for k, v in rows)
    return ("<div class=lockbox><h3 style='margin:.1rem 0 .5rem;font-size:.85rem;"
            "text-transform:uppercase;letter-spacing:.06em;color:var(--muted)'>"
            "Termination switches</h3>" + body +
            f"<p class=sub style='margin:.5rem 0 0'>{html.escape(why)}</p></div>"), allowed


def _progress_html():
    doc = terminate_worker.read_progress()
    if not doc:
        return ""
    cls = "banner"
    if doc.get("done"):
        cls = "banner good" if doc.get("ok") else "banner bad"
    steps = "".join(
        f"<li>{html.escape(s.get('phase', ''))}"
        + (f" — {html.escape(s.get('detail', ''))}" if s.get("detail") else "")
        + "</li>" for s in doc.get("steps", []))
    extra = ""
    if doc.get("dry_run"):
        extra += ("<p><b>DRY RUN</b> — the destroy request is being prepared but NOT "
                  "sent. The instance keeps running and keeps billing.</p>")
    if doc.get("terminate", {}).get("attempts"):
        rows = "".join(
            "<li><code>%s %s</code> → HTTP %s</li>" % (
                html.escape(str(a.get("method"))), html.escape(str(a.get("url"))),
                html.escape(str(a.get("status"))))
            for a in doc["terminate"]["attempts"])
        extra += f"<p class=sub>Provider calls:</p><ul>{rows}</ul>"
    er = doc.get("erase_result")
    if er:
        extra += ("<p class=sub>Erased %s file(s), %.1f MiB overwritten%s.</p>" % (
            er.get("erased", 0), er.get("bytes", 0) / 2**20,
            f", {len(er.get('failed', []))} failed" if er.get("failed") else ""))
    if doc.get("unknown_large"):
        extra += ("<p class=sub><b>Not erased:</b> %d large file(s) under the model dir "
                  "could not be told apart from the public checkpoint. They are listed "
                  "in the progress JSON.</p>" % len(doc["unknown_large"]))
    head = ("Termination finished" if doc.get("done") else "Termination in progress")
    return (f"<div class='{cls}'><b>{head}</b> — phase: "
            f"<code>{html.escape(doc.get('phase', ''))}</code>"
            f"<p>{html.escape(doc.get('detail', ''))}</p>{extra}"
            f"<details><summary>steps</summary><ul>{steps}</ul></details></div>")


_PROBE_CACHE = {"ts": 0, "state": "unknown", "detail": ""}


def credential_probe(p):
    """Ask the provider whether the credential is even valid, before the user
    types an instance id. Uses an UNARMED transport: the check is a read, and
    the page never needs the ability to destroy anything. Cached, because it is
    a network round trip on a page that gets polled."""
    if os.environ.get("TERMINATE_PROBE", "1") == "0":
        return "unknown", "credential pre-check disabled (TERMINATE_PROBE=0)"
    now = time.time()
    if now - _PROBE_CACHE["ts"] < 60:
        return _PROBE_CACHE["state"], _PROBE_CACHE["detail"]
    try:
        state, detail = p.probe(prov.HttpTransport(timeout=8))
    except Exception as e:
        state, detail = "unknown", f"the credential check could not run ({e})"
    _PROBE_CACHE.update({"ts": now, "state": state, "detail": detail})
    return state, detail


def render_terminate(tok: str, secure: bool, banner=None, banner_cls="") -> bytes:
    p = prov.get()
    desc = p.describe()
    switch_html, allowed = _switch_html()
    tok_q = html.escape(tok, quote=True)
    dry = gc.env_flag(os.environ, "TERMINATE_DRY_RUN", False)
    expected = desc["instance_id"] or "TERMINATE"

    parts = ["<!doctype html><html><head><title>Terminate this instance</title>"
             "<meta name=viewport content='width=device-width,initial-scale=1'>",
             STYLE, CONFIG_STYLE, TERMINATE_STYLE, "</head><body><div class=wrap>",
             "<header class=hero><h1><b>Terminate</b> this instance</h1>"
             f"<span class=sub><a href='/?token={tok_q}'>&larr; status</a> &middot; "
             f"<a href='/config?token={tok_q}'>configuration</a></span></header>"]
    if banner:
        parts.append(f"<div class='banner {banner_cls}'>{banner}</div>")
    progress = _progress_html()
    if progress:
        parts.append(progress)
    parts.append(switch_html)

    parts.append(
        "<div class=grid>"
        "<div class=card><h3>Provider</h3><div class=v>%s</div>"
        "<div class=why>%s</div></div>"
        "<div class=card><h3>Instance</h3><div class=v><code>%s</code></div>"
        "<div class=why>%s</div></div></div>" % (
            html.escape(desc["label"]),
            html.escape(desc["reason"]),
            html.escape(desc["instance_id"] or "(unknown)"),
            html.escape(desc["key_source"] or "no credential found")))

    if desc["supported"] and desc["ready"]:
        pstate, pdetail = credential_probe(p)
        cls = {"ok": "banner good", "rejected": "banner bad"}.get(pstate, "banner")
        parts.append(f"<div class='{cls}'><b>Credential check:</b> "
                     f"{html.escape(pdetail)}</div>")
    for w in desc.get("warnings", []):
        parts.append(f"<div class='banner bad'>{html.escape(w)}</div>")

    if not desc["supported"]:
        ev = prov.detection_evidence()
        parts.append(
            "<div class=danger><h2>Termination is not supported here</h2>"
            "<p>This image could not identify the cloud provider it is running on, so "
            "it will not try to destroy anything. <b>Terminate from your provider's "
            "dashboard</b> — that is what actually stops the billing.</p>"
            "<p class=sub>If you know the provider, relaunch with "
            "<code>TERMINATE_PROVIDER=vastai</code> or <code>=runpod</code>.</p>"
            "<details><summary>what the detector saw</summary><pre>%s</pre></details>"
            "</div>" % html.escape(json.dumps(ev, indent=1)))
        parts.append("</div></body></html>")
        return "".join(parts).encode()

    parts.append("<div class=danger><h2>This is irreversible</h2>"
                 "<p>Destroying the instance <b>cannot be undone</b>. There is no "
                 "trash, no grace period, and no support ticket that brings it back.</p>"
                 "<p><b>Destroyed:</b></p><ul>"
                 + "".join(f"<li>{html.escape(d)}</li>" for d in desc["destroys"])
                 + "</ul><p><b>Survives:</b></p><ul>"
                 + "".join(f"<li>{html.escape(s)}</li>" for s in desc["survives"])
                 + f"</ul><p class=sub>{html.escape(desc['billing'])}</p></div>")

    if dry:
        parts.append("<div class=banner><b>TERMINATE_DRY_RUN=1</b> — the destroy "
                     "request will be prepared and shown, but not sent.</div>")

    if not allowed:
        parts.append("<p class=sub>The form below is disabled while the switches "
                     "above refuse termination.</p>")

    dis = "" if allowed else " disabled"
    parts.append(
        f"<form method=post action='/terminate?token={tok_q}'>"
        f"<input type=hidden name=token value='{tok_q}'>"
        "<h2>Optional: erase this session first</h2>"
        f"<p><label><input type=checkbox name=erase value=1{dis}> "
        "<b>Securely erase session data before destroying</b> (default off)</label></p>"
        "<p class=sub>Overwrites and unlinks the API key, TLS private key, config "
        "state, every log this template writes (prompts included, where request "
        "logging put them), shell history, SSH material, provider and Hugging Face "
        "credentials, and anything you added under the model dir. "
        "<b>The public model weights are deliberately NOT erased</b> — the checkpoint "
        "is downloadable by anyone, so overwriting 332 GB hides nothing and would take "
        "far longer than you have. "
        "<b>Limits:</b> on SSDs with wear levelling, on overlay/copy-on-write "
        "filesystems, and on network volumes, an overwrite does not guarantee the old "
        "bytes are unreachable; provider snapshots and the instance console log in "
        "your dashboard are outside our reach entirely.</p>"
        f"<p><label><input type=checkbox name=ram value=1{dis}> also drop the page "
        "cache and overwrite free RAM</label><br>"
        f"<label><input type=checkbox name=vram value=1{dis}> also zero GPU memory "
        "(after the engine stops)</label></p>"
        "<h2>Confirm</h2>"
        f"<p class=sub>Type <code>{html.escape(expected)}</code> exactly — the "
        "instance id. This is deliberately not a one-click action.</p>"
        f"<p><input class=confirm name=confirm autocomplete=off "
        f"placeholder='{html.escape(expected, quote=True)}'{dis}></p>"
        f"<p><label><input type=checkbox name=understand value=1{dis}> I understand "
        "this destroys the instance and everything on its disks, permanently.</label></p>"
        f"<p><button class=destroy type=submit{dis}>Destroy this instance</button></p>"
        "</form>")

    parts.append(
        f"<div class=lockbox><h2 style='margin-top:.2rem'>Lock this instance instead</h2>"
        "<p class=sub>Locking is one-way. It refuses termination from this page for the "
        "life of this container, for anyone holding this token — including you. It "
        "clears only by restarting the container with <code>TERMINATE_LOCKED</code> "
        "unset, which needs provider-dashboard access.</p>"
        f"<form method=post action='/terminate/lock?token={tok_q}'>"
        f"<input type=hidden name=token value='{tok_q}'>"
        "<p><label><input type=checkbox name=understand value=1> I understand this "
        "cannot be undone from this page.</label></p>"
        "<p><button type=submit name=action value=disable>Disable the terminate "
        "control</button> <button type=submit name=action value=lock>Lock termination "
        "(hard)</button></p></form></div>")

    parts.append("</div></body></html>")
    return "".join(parts).encode()


def start_termination(form) -> tuple:
    """Validate every gate, then hand off to a detached worker.

    -> (started, message, css_class). Returns without acting on any failure."""
    allowed, why = gc.termination_allowed()
    if not allowed:
        return False, html.escape(why), "bad"
    if form.get("understand", [""])[0] != "1":
        return False, ("Nothing was done: you did not tick the box confirming you "
                       "understand this is permanent."), "bad"
    p = prov.get()
    ok, reason = p.ready()
    if not ok:
        return False, html.escape(reason), "bad"
    expected = p.instance_id() or "TERMINATE"
    typed = (form.get("confirm", [""])[0] or "").strip()
    if typed != expected:
        return False, ("Nothing was done: the confirmation text did not match. Type "
                       f"<code>{html.escape(expected)}</code> exactly."), "bad"
    with _terminate_lock:
        doc = terminate_worker.read_progress()
        if doc and not doc.get("done"):
            return False, "A termination is already in progress.", ""
        opts = ["--confirm", typed]
        if form.get("erase", [""])[0] == "1":
            opts.append("--erase")
            if form.get("ram", [""])[0] == "1":
                opts.append("--ram")
            if form.get("vram", [""])[0] == "1":
                opts.append("--vram")
        try:
            os.makedirs(gc.runtime_dir(), exist_ok=True)
            logf = open(os.path.join(gc.runtime_dir(), "terminate.log"), "ab")
            subprocess.Popen(
                [sys.executable, os.path.join(SCRIPTS_DIR, "terminate_worker.py")] + opts,
                stdout=logf, stderr=logf, start_new_session=True, close_fds=True)
        except Exception as e:
            return False, ("Could not start the termination worker: "
                           + html.escape(str(e))), "bad"
        return True, ("Termination started. This page now shows its progress; do not close "
                      "it until the outcome is reported."), ""


def render(secure: bool, tok: str = "") -> bytes:
    st = status()
    endpoint = st.get("endpoint", "")
    weights = weights_state()
    engine = engine_state()
    if engine == "serving":
        weights = "ready"  # a serving engine is proof enough (pre-marker volumes)
    serving = engine == "serving"

    def card(label, value, ok):
        cls = "ok" if ok else "busy"
        return (f"<div class=card><h3>{html.escape(label)}</h3>"
                f"<div class=v><span class='pill {cls}'></span>"
                f"<span class={cls}>{html.escape(value)}</span></div></div>")

    real0 = st.get("api_key", "")
    dep = deployment()
    chat_ok = bool(secure and TOKEN and real0 and endpoint and serving)
    wrap_cls = "wrap wide" if chat_ok else "wrap"
    subtitle = " &middot; ".join(html.escape(c) for c in dep["chips"])
    parts = [page_head(dep["title"]),
             f'<div class="{wrap_cls}"><div class=layout><main>',
             f"<header class=hero><h1><b>{html.escape(dep['model'] or 'vLLM')}</b> "
             f"turnkey</h1><span class=sub>{subtitle}</span></header>",]
    if not dep.get("tested", True):
        parts.append("<div class=card style='border-color:#e5484d'><h3>Residual "
                     "qualification</h3><div class=v>" + html.escape(dep["family"]) +
                     "</div><div class=sub>This exact full-size checkpoint/profile "
                     "has not completed the production-scale matrix. Treat a clean "
                     "boot as the start of model-specific validation.</div>"
                     "</div>")
    parts += [
             "<div class=grid>",
             card("Weights", weights, weights == "ready"),
             card("TLS / DNS", st.get("tls", "not configured"),
                  st.get("tls", "").startswith("https")),
             card("Engine", engine, serving),
             card("DRAM KV offload", st.get("offload", "off"),
                  st.get("offload", "off") != "off"),
             "</div>"]
    parts.append(
        "<div class=card><h3>Deployment</h3>"
        "<div class=v>%s</div>"
        "<div class=sub style='margin-top:.3rem'>%s%s%s</div></div>" % (
            html.escape(dep["family"] or "unknown family"),
            ("repo <code>%s</code>" % html.escape(dep.get("repo", "")))
            if dep.get("repo") else "",
            (" &middot; %s GPU(s)%s" % (html.escape(str(dep["gpus"])),
                                        (" " + html.escape(dep["gpu_name"]))
                                        if dep["gpu_name"] else ""))
            if dep.get("gpus") else "",
            (" &middot; provider %s" % html.escape(dep["provider"]))
            if dep.get("provider") and dep["provider"] != "unknown" else ""))
    parts.append(render_soul_card(tok))
    if gc is not None:
        # "Engine: serving" above is a LIVENESS statement. Correctness is a
        # separate card on purpose — every silent-corruption configuration this
        # appliance has met answered /health and short prompts perfectly.
        head, ok, _detail = correctness()
        parts.append("<div class=grid>" + card("Correctness", head, ok) + "</div>")
        if config_enabled():
            mode = gc.apply_state().get("mode", "steady")
            note = ("<b>The last change you applied was rolled back</b> — the previous "
                    "known-good configuration is running. " if mode == "rolled-back" else "")
            parts.append(
                f'<div class=card><h3>Configuration</h3><div class=v>{note}'
                f'<a href="/config?token={html.escape(tok, quote=True)}">'
                "Change model, draft, context, concurrency &rarr;</a></div>"
                "<div class=sub style='margin-top:.3rem'>Edits are written to a JSON "
                "state file on the volume and applied by restarting vLLM only.</div>"
                "</div>")
        else:
            parts.append("<p class=sub>Set <code>OPEN_BUTTON_TOKEN</code> in the template "
                         "environment to unlock the self-service configuration editor.</p>")
    if endpoint:
        ep = html.escape(endpoint, quote=True)
        real = st.get("api_key", "")  # from the root-only status file
        key = real if (secure and TOKEN and real) else "<paste API key from instance logs>"
        tok_esc = html.escape(tok, quote=True)
        parts.append(f'<div class=card><h3>OpenAI-compatible endpoint</h3>'
                     f'<div class=v><a href="{ep}/v1/models"><code>{ep}/v1</code></a></div>'
                     f'<div class=sub style="margin-top:.4rem">'
                     f'<a href="{ep}/metrics">Prometheus /metrics</a>')
        if not key.startswith("<"):
            parts.append(f' &middot; <a href="/chat?token={tok_esc}"><b>Quick chat &rarr;</b></a>')
        if terminate_available():
            parts.append(f' &middot; <a href="/terminate?token={tok_esc}">'
                         'Terminate instance</a>')
        parts.append('</div></div>')
        if key.startswith("<"):
            where = {
                "vastai": "the instance logs (vast console &rarr; Logs, look for "
                          "<code>API KEY</code>)",
                "runpod": "the pod's container logs. <b>RunPod exposes no console log "
                          "view and no CLI log command</b>, so if you cannot see them, "
                          "open this page with the token from the pod environment "
                          "(<code>OPEN_BUTTON_TOKEN</code>) &mdash; it is shown here "
                          "once the token check passes",
            }.get(dep.get("provider"), "the container logs, or set "
                                       "<code>VLLM_API_KEY</code> yourself")
            parts.append(f"<p class=sub>The API key is printed in {where}.</p>")
        if serving:
            parts.append(METRICS_SECTION.substitute(
                ep_js=js_literal(endpoint), key_js=js_literal(key)))
        parts.append("<h2>Client configs</h2>")
        for name, where, body in SNIPPETS:
            filled = Template(body).substitute(
                ep=endpoint, key=key, model=dep["model"] or "model",
                ctx=dep.get("context") or 32768, output=dep.get("output") or 8192,
                input=dep.get("input") or "[text]")
            parts.append(f"<details><summary>{html.escape(name)}</summary>"
                         f"<p class=sub><code>{html.escape(where)}</code></p>"
                         f"<pre>{html.escape(filled)}</pre></details>")
        parts.append(
            '<h2>Quick test <button id=copybtn onclick="copyQT()" '
            'style="vertical-align:middle">copy</button></h2>'
            f"<pre id=qt>curl -H \"Authorization: Bearer {html.escape(key)}\" {ep}/v1/models</pre>"
            "<script>function copyQT(){navigator.clipboard.writeText("
            "document.getElementById('qt').textContent.trim()).then(()=>{"
            "const b=document.getElementById('copybtn');b.textContent='copied!';"
            "setTimeout(()=>{b.textContent='copy'},1500)})}</script>")
        if serving:
            parts.append('<h2>Model</h2><div class=card>'
                         '<pre id=modeljson style="max-height:16rem;margin:0;border:none">'
                         'loading /v1/models&hellip;</pre></div>')
    notes = boot_notes()
    if notes:
        parts.append(
            "<h2>Boot log highlights</h2><div class=card><pre style='margin:0;"
            "border:none;max-height:14rem'>" + html.escape("\n".join(notes)) +
            "</pre></div><p class=sub>Warnings the entrypoint emitted before the "
            "engine came up. This page carries them because some providers expose "
            "no container console at all.</p>")
    if not serving:
        # keep boot status fresh; once serving, the dashboard polls instead
        parts.append("<script>setTimeout(function(){location.reload()},20000)</script>"
                     "<p class=sub>Auto-refreshing every 20 s while booting.</p>")
    parts.append("</main>")
    if chat_ok:
        parts.append("<aside class=chatpane><h2>Quick chat</h2>" + CHAT_WIDGET +
                     "</aside>" + Template(CHAT_JS).substitute(
                         ep_js=js_literal(endpoint), key_js=js_literal(real0),
                         model_js=js_literal(dep["model"] or "model"),
                         output=dep.get("output") or 8192))
    parts.append("</div></div></body></html>")
    return "".join(parts).encode()


class DualProtocolServer(ThreadingHTTPServer):
    """Peek the first byte of each connection: TLS handshake (0x16) or plain HTTP."""

    # Handler threads are daemon so a stalled client cannot keep the process
    # alive indefinitely; each accepted socket gets a recv timeout below.
    daemon_threads = True
    timeout = LANDING_TIMEOUT

    def get_request(self):
        sock, addr = self.socket.accept()
        sock.settimeout(LANDING_TIMEOUT)
        try:
            first = sock.recv(1, socket.MSG_PEEK)
        except (socket.timeout, OSError):
            sock.close()
            raise
        ctx = ssl_ctx()
        if ctx and first == b"\x16":
            try:
                sock = ctx.wrap_socket(sock, server_side=True)
            except (socket.timeout, ssl.SSLError):
                sock.close()
                raise OSError("TLS handshake failed")
        return sock, addr


GET_PATHS = ("/", "/chat", "/config", "/config/export", "/config/status",
             "/soul", "/soul/status", "/soul/journal",
             "/terminate", "/terminate/status")
POST_PATHS = ("/config/apply", "/config/import", "/config/reset", "/config/restart",
              "/soul/config", "/terminate", "/terminate/lock")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the vast console log quiet
        pass

    def end_headers(self):
        # Pages contain bearer credentials and capability tokens. Keep them out
        # of browser/proxy caches, referrers, MIME sniffing and framing contexts.
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()

    # -- helpers ---------------------------------------------------------
    def _send(self, body: bytes, ctype="text/html; charset=utf-8", code=200, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _config_guard(self, tok):
        """The editor changes what the box serves; it needs the same token the
        Open button carries, and it needs the shared registry to be importable."""
        if gc is None:
            self.send_error(503, "config editor unavailable (glm_config import failed)")
            return False
        if not TOKEN or not hmac.compare_digest(tok.encode("utf-8", "replace"),
                                              TOKEN.encode("utf-8")):
            self.send_error(403, "the config editor requires OPEN_BUTTON_TOKEN")
            return False
        return True

    def _terminate_guard(self, tok):
        """Termination is never reachable unauthenticated, and never reachable
        at all unless the shared modules are importable."""
        if not self._config_guard(tok):
            return False
        if not terminate_available():
            self.send_error(503, "termination support is unavailable in this image")
            return False
        return True

    def _read_form(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > 1 << 20:
            return {}
        try:
            return parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
        except (socket.timeout, OSError):
            # A stalled client (LANDING_TIMEOUT) or a broken pipe must not
            # take the page down; an empty form simply fails the auth check.
            return {}

    def _redirect(self, path):
        self.send_response(303)
        self.send_header("Location", path)
        self.end_headers()

    # -- POST: apply / import / reset / restart --------------------------
    def do_POST(self):
        url = urlparse(self.path)
        if url.path not in POST_PATHS:
            self.send_error(404)
            return
        form = self._read_form()
        tok = form.get("token", [""])[0] or parse_qs(url.query).get("token", [""])[0]
        if not self._config_guard(tok):
            return
        secure = is_secure_connection(self.connection)
        try:
            if url.path == "/config/apply":
                values, errs = _form_values(form)
                if errs:
                    body = render_config(tok, secure, "Nothing was applied.", "bad", errs)
                else:
                    ok, findings, msg = apply_values(values)
                    body = render_config(tok, secure, msg, "good" if ok else "bad",
                                         [] if ok else findings)
            elif url.path == "/config/import":
                doc = form.get("doc", [""])[0]
                try:
                    parsed = json.loads(doc)
                    values = parsed.get("values", parsed) if isinstance(parsed, dict) else None
                    if not isinstance(values, dict):
                        raise ValueError("expected a JSON object of knob values")
                except ValueError as e:
                    body = render_config(tok, secure, f"Import failed: {html.escape(str(e))}",
                                         "bad")
                else:
                    coerced, errs = _coerce_imported_values(values)
                    if errs:
                        body = render_config(tok, secure, "Nothing was imported.",
                                             "bad", errs)
                    else:
                        ok, findings, msg = apply_values(coerced)
                        body = render_config(tok, secure, msg, "good" if ok else "bad",
                                             [] if ok else findings)
            elif url.path == "/config/reset":
                try:
                    os.remove(gc.p_state())
                except OSError:
                    pass
                gc.set_apply_state("trial", detail="reset to template env + defaults")
                os.makedirs(gc.runtime_dir(), exist_ok=True)
                with open(gc.p_restart_flag(), "w") as f:
                    f.write("reset\n")
                body = render_config(tok, secure,
                                     "State file removed; restarting on the template "
                                     "environment and the built-in defaults.", "good")
            elif url.path == "/config/restart":
                os.makedirs(gc.runtime_dir(), exist_ok=True)
                with open(gc.p_restart_flag(), "w") as f:
                    f.write("manual\n")
                body = render_config(tok, secure, "Restart requested.", "good")
            elif url.path == "/soul/config":
                if soul is None:
                    self.send_error(503, "SOUL support is unavailable")
                    return
                values = soul.load_override()
                for name in (*soul.FIELDS, "timezone"):
                    if name in form:
                        values[name] = form[name][0]
                effective = soul.write_override(values)
                body = render_soul(
                    tok, secure,
                    f"SOUL settings saved. Effective autonomy level is "
                    f"{effective['autonomyLevel']}.", "good")
            elif url.path == "/terminate":
                if not self._terminate_guard(tok):
                    return
                started, msg, cls = start_termination(form)
                body = render_terminate(tok, secure, msg, cls if cls else
                                        ("good" if started else "bad"))
            else:  # /terminate/lock — the runtime ratchet, tighten-only
                if not self._terminate_guard(tok):
                    return
                if form.get("understand", [""])[0] != "1":
                    body = render_terminate(tok, secure, "Nothing changed: tick the "
                                            "box first.", "bad")
                else:
                    action = form.get("action", [""])[0]
                    if action == "lock":
                        gc.tighten(locked=True, reason="locked from the landing page")
                        msg = ("Termination is now LOCKED for the life of this "
                               "container. This page cannot unlock it.")
                    else:
                        gc.tighten(enabled=False, reason="disabled from the landing page")
                        msg = ("The terminate control is now disabled. It cannot be "
                               "re-enabled from this page.")
                    body = render_terminate(tok, secure, msg, "good")
        except Exception as e:  # never take the landing page down on a bad form
            err = "Internal error: " + html.escape(str(e))
            if url.path.startswith("/soul"):
                body = render_soul(tok, secure, err, "bad")
            elif url.path.startswith("/terminate"):
                body = render_terminate(tok, secure, err, "bad")
            else:
                body = render_config(tok, secure, err, "bad")
        self._send(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path not in GET_PATHS:
            self.send_error(404)
            return
        tok = parse_qs(url.query).get("token", [""])[0]
        # Strip CR/LF so a crafted token can never inject header fields into
        # the plain->HTTPS redirect Location below.
        tok = tok.replace("\r", "").replace("\n", "")
        if TOKEN and not hmac.compare_digest(tok.encode("utf-8", "replace"),
                                           TOKEN.encode("utf-8")):
            self.send_error(403, "bad token (use the vast console Open button)")
            return
        secure = is_secure_connection(self.connection)
        hostport = status().get("https_hostport", "")
        if not secure and ssl_ctx() and hostport:
            # Upgrade the Open button's plain-HTTP hit to the TLS view of this page
            q = f"?token={tok}" if tok else ""
            self.send_response(302)
            self.send_header("Location", f"https://{hostport}{url.path}{q}")
            self.end_headers()
            return
        if url.path == "/chat":
            st = status()
            key = st.get("api_key", "")
            # the chat page embeds the key in JS: TLS (or trusted-LAN) + token gate
            if not (secure and TOKEN and key and st.get("endpoint")):
                self.send_error(403, "chat needs TLS + token gate + a running endpoint")
                return
            dep = deployment()
            body = CHAT_PAGE.substitute(
                                        ep_js=js_literal(st["endpoint"]),
                                        key_js=js_literal(key),
                                        model_js=js_literal(dep["model"] or "model"),
                                        model_html=html.escape(dep["model"] or "model"),
                                        output=dep.get("output") or 8192,
                                        token_html=html.escape(tok, quote=True)).encode()
        elif url.path == "/config":
            if not self._config_guard(tok):
                return
            self._send(render_config(tok, secure))
            return
        elif url.path == "/soul":
            if not self._config_guard(tok):
                return
            before = parse_qs(url.query).get("before", [""])[0]
            self._send(render_soul(tok, secure, before=before))
            return
        elif url.path == "/soul/status":
            if not self._config_guard(tok):
                return
            self._send(json.dumps(soul_status(), ensure_ascii=False, indent=1).encode(),
                       "application/json")
            return
        elif url.path == "/soul/journal":
            if not self._config_guard(tok):
                return
            if soul is None:
                self.send_error(503, "SOUL support is unavailable")
                return
            query = parse_qs(url.query)
            try:
                limit = max(1, min(int(query.get("limit", ["20"])[0]), 50))
            except ValueError:
                limit = 20
            before = query.get("before", [""])[0]
            self._send(json.dumps(soul.read_journal(limit=limit, before=before),
                                  ensure_ascii=False).encode(), "application/json")
            return
        elif url.path == "/config/export":
            if not self._config_guard(tok):
                return
            effective, sources, _notes = gc.resolve()
            try:
                values = gc.load_state_file()
            except Exception:
                values = {}
            doc = json.dumps({
                "exported_at": gc.utcnow_iso(),
                "note": ("'values' is what an import applies: only the knobs that "
                         "override the defaults and the template environment. "
                         "'effective' and 'sources' are informational."),
                "values": values, "effective": effective, "sources": sources,
            }, indent=1).encode()
            self._send(doc, "application/json",
                       extra=(("Content-Disposition",
                               "attachment; filename=glm52-config.json"),))
            return
        elif url.path == "/terminate":
            if not self._terminate_guard(tok):
                return
            self._send(render_terminate(tok, secure))
            return
        elif url.path == "/terminate/status":
            if not self._terminate_guard(tok):
                return
            allowed, why = gc.termination_allowed()
            self._send(json.dumps({
                "switches": gc.read_switches(), "allowed": allowed, "reason": why,
                "provider": prov.get().describe(),
                "progress": terminate_worker.read_progress(),
            }, indent=1).encode(), "application/json")
            return
        elif url.path == "/config/status":
            if not self._config_guard(tok):
                return
            head, ok, detail = correctness()
            self._send(json.dumps({
                "apply_state": gc.apply_state(), "verify": verify_last(),
                "correctness": {"headline": head, "ok": ok, "detail": detail},
                "phase": status().get("phase", ""),
            }, indent=1).encode(), "application/json")
            return
        else:
            body = render(secure, tok)
        self._send(body)


if __name__ == "__main__":
    # 1111 is the port the vast template maps and the Open button hits;
    # LANDING_PORT exists so the page can be exercised somewhere else.
    DualProtocolServer(("0.0.0.0", int(os.environ.get("LANDING_PORT", "1111"))),
                       Handler).serve_forever()
