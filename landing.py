#!/usr/bin/env python3
"""Vast.ai/Runpod landing page — dual-protocol (TLS + plain) on :1111.

Started first thing in entrypoint.sh so status is visible from the very first
boot minute. Each connection's first byte is peeked: 0x16 = TLS handshake,
else plain HTTP. Once a cert exists (lazy-loaded mid-boot), plain hits are
302-redirected to the HTTPS view. Boot state comes from the status file the
entrypoint rewrites at each milestone, plus live probes (download size,
/health). The API key, filled-in client configs (oh-my-pi, opencode, Claude
Code, Codex), live metrics dashboard and /chat render only over TLS with a
validated OPEN_BUTTON_TOKEN — over a trusted HTTPS reverse proxy, or over
plain HTTP if LANDING_ALLOW_INSECURE=1 (trusted-LAN/demo hosts without a
cert). The dashboard scrapes vLLM's
/metrics from the browser (vLLM CORS is open) — no server-side proxying.
Vast uses OPEN_BUTTON_PORT=1111 + '-p 1111:1111'; Runpod uses 1111/http.
"""
import html
import json
import os
import socket
import ssl
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from string import Template
from urllib.parse import parse_qs, urlparse

TOKEN = os.environ.get("OPEN_BUTTON_TOKEN", "")
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models/current")
STATUS_FILE = os.environ.get("STATUS_FILE", "/tmp/model-turnkey-boot-status.json")
MODEL_DISPLAY_NAME = os.environ.get("MODEL_DISPLAY_NAME", "Local model")
MODEL_ID = os.environ.get("LANDING_MODEL_ID", MODEL_DISPLAY_NAME)
MODEL_PROFILE = os.environ.get("MODEL_PROFILE", "custom")
MODEL_CONTEXT_WINDOW = int(os.environ.get("MODEL_CONTEXT_WINDOW", "32768"))
MODEL_OUTPUT_LIMIT = int(os.environ.get("MODEL_OUTPUT_LIMIT", "8192"))
LANDING_FEATURES = os.environ.get(
    "LANDING_FEATURES", "OpenAI-compatible · self-hosted"
)
ALLOW_INSECURE = os.environ.get("LANDING_ALLOW_INSECURE", "0") == "1"
TRUST_PROXY_HTTPS = os.environ.get("LANDING_TRUST_PROXY_HTTPS", "0") == "1"
API_PORT = os.environ.get("LANDING_API_PORT", "8000")
WEIGHTS_TOTAL_GIB = float(os.environ.get("MODEL_DOWNLOAD_GIB", "0"))

_ssl_ctx = None


def is_secure_connection(connection) -> bool:
    """Whether secrets may be rendered on this request's transport."""
    return (
        isinstance(connection, ssl.SSLSocket)
        # Runpod's documented HTTP endpoint is HTTPS-only, but its proxy
        # header contract is not documented. Trust it only when the
        # entrypoint explicitly selected the Runpod dashboard proxy.
        or TRUST_PROXY_HTTPS
        or ALLOW_INSECURE
    )


def ssl_ctx():
    """Lazy: the cert is issued minutes into the boot; pick it up when it lands."""
    global _ssl_ctx
    if _ssl_ctx is None:
        st = status()
        crt, key = st.get("cert", ""), st.get("keyfile", "")
        if crt and key and os.path.isfile(crt):
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(crt, key)
            _ssl_ctx = ctx
    return _ssl_ctx


def status() -> dict:
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def weights_state() -> str:
    if os.path.isfile(os.path.join(MODEL_DIR, ".download-complete")):
        return "ready"
    done = 0
    for root, _dirs, files in os.walk(MODEL_DIR):
        for f in files:
            try:
                done += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    downloaded = done / 2**30
    if WEIGHTS_TOTAL_GIB > 0:
        return f"downloading — {downloaded:.0f} of ~{WEIGHTS_TOTAL_GIB:g} GiB"
    return f"downloading — {downloaded:.0f} GiB"


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
    ("oh-my-pi / pi", "~/.pi/agent/models.json", """{
  "providers": {
    "turnkey": {
      "baseUrl": "$ep/v1",
      "api": "openai-completions",
      "apiKey": "$key",
      "models": [
        {"id": "$model", "name": "$display", "contextWindow": $context}
      ]
    }
  }
}"""),
    ("opencode", "opencode.json (project or ~/.config/opencode/)", """{
  "$$schema": "https://opencode.ai/config.json",
  "provider": {
    "turnkey": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "$display",
      "options": {"baseURL": "$ep/v1", "apiKey": "$key"},
      "models": {
        "$model": {"name": "$display", "limit": {"context": $context, "output": $output}}
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
name = "$display"
base_url = "$ep/v1"
env_key = "TURNKEY_API_KEY"
wire_api = "chat"

# then:  export TURNKEY_API_KEY="$key" """),
]

# Shared look. Literal CSS only (no stray $): safe inside plain strings.
STYLE = """<style>
:root{color-scheme:light;--bg:#f2f5fb;--bg2:#e9eef8;--card:rgba(255,255,255,.82);
--card-solid:#fff;--fg:#152033;--muted:#66728a;--faint:#8994aa;
--line:rgba(66,85,119,.16);--line-strong:rgba(66,85,119,.28);
--accent:#315efb;--accent2:#8b5cf6;--cyan:#06a8c7;--ok:#0c9b67;--busy:#d57a08;
--danger:#d44255;--shadow:0 18px 50px rgba(34,51,84,.10);
--shadow-sm:0 6px 18px rgba(34,51,84,.07);
--mono:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace}
@media(prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#080c14;--bg2:#0e1523;
--card:rgba(17,24,39,.82);--card-solid:#111827;--fg:#edf2ff;--muted:#9ba8c0;
--faint:#71809a;--line:rgba(148,163,184,.16);--line-strong:rgba(148,163,184,.30);
--accent:#6f8fff;--accent2:#ae7cff;--cyan:#33c5df;--ok:#35d399;--busy:#f5ad4f;
--danger:#fb7185;--shadow:0 22px 60px rgba(0,0,0,.34);
--shadow-sm:0 8px 22px rgba(0,0,0,.22)}}
*{box-sizing:border-box}
html{min-height:100%}
body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
background:
radial-gradient(circle at 12% -5%,color-mix(in srgb,var(--accent) 19%,transparent),transparent 29rem),
radial-gradient(circle at 92% 8%,color-mix(in srgb,var(--accent2) 15%,transparent),transparent 25rem),
linear-gradient(155deg,var(--bg),var(--bg2));background-attachment:fixed;color:var(--fg);
margin:0;line-height:1.5;min-height:100vh}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.32;
background-image:linear-gradient(var(--line) 1px,transparent 1px),
linear-gradient(90deg,var(--line) 1px,transparent 1px);background-size:36px 36px;
mask-image:linear-gradient(to bottom,black,transparent 68%)}
.wrap{position:relative;max-width:68rem;margin:0 auto;padding:1.25rem 1rem 4rem}
.layout{display:grid;gap:1.25rem;grid-template-columns:minmax(0,1fr)}
main{min-width:0}
header.hero{display:flex;align-items:center;gap:1rem;margin:.35rem 0 1.2rem;
padding:1.15rem 1.25rem;border:1px solid var(--line);border-radius:20px;
background:linear-gradient(135deg,var(--card),color-mix(in srgb,var(--card-solid) 62%,transparent));
box-shadow:var(--shadow-sm);backdrop-filter:blur(18px)}
.brandmark{position:relative;display:grid;place-items:center;flex:0 0 3rem;width:3rem;height:3rem;
border-radius:14px;background:linear-gradient(145deg,var(--accent),var(--accent2));
box-shadow:0 10px 24px color-mix(in srgb,var(--accent) 28%,transparent)}
.brandmark:before{content:"";width:1.18rem;height:1.18rem;border:2px solid rgba(255,255,255,.92);
border-radius:5px;box-shadow:inset 0 0 0 3px rgba(255,255,255,.16)}
.brandmark:after{content:"";position:absolute;width:.34rem;height:.34rem;border-radius:50%;
background:#fff;box-shadow:.52rem 0 rgba(255,255,255,.8),-.52rem 0 rgba(255,255,255,.8)}
.hero-copy{min-width:0;flex:1}
.eyebrow{font-size:.68rem;line-height:1.2;text-transform:uppercase;letter-spacing:.13em;
font-weight:750;color:var(--accent)}
header.hero h1{margin:.12rem 0 0;font-size:clamp(1.25rem,3vw,1.75rem);
line-height:1.2;letter-spacing:-.035em;overflow-wrap:anywhere}
header.hero h1 b{font-weight:760}
.sub{color:var(--muted);font-size:.88rem}
header.hero .sub{display:block;margin-top:.18rem}
.hero-state{display:flex;align-items:center;gap:.45rem;white-space:nowrap;font-size:.77rem;
font-weight:700;color:var(--muted);padding:.42rem .68rem;border:1px solid var(--line);
border-radius:999px;background:color-mix(in srgb,var(--card-solid) 50%,transparent)}
h2{font-size:1rem;margin:1.7rem 0 .65rem;letter-spacing:-.015em}
.grid{display:grid;gap:.75rem;grid-template-columns:repeat(auto-fit,minmax(12rem,1fr));margin:.75rem 0}
.card{position:relative;background:var(--card);border:1px solid var(--line);border-radius:16px;
padding:1rem 1.05rem;box-shadow:var(--shadow-sm);backdrop-filter:blur(16px)}
.status-card{overflow:hidden;min-height:6.2rem;transition:transform .18s ease,border-color .18s ease}
.status-card:hover{transform:translateY(-2px);border-color:var(--line-strong)}
.status-card:after{content:"";position:absolute;right:-1.2rem;bottom:-1.8rem;width:4.5rem;height:4.5rem;
border-radius:50%;background:color-mix(in srgb,var(--accent) 8%,transparent)}
.card h3{margin:0 0 .45rem;font-size:.67rem;text-transform:uppercase;
letter-spacing:.105em;color:var(--muted);font-weight:750}
.card .v{font-size:.98rem;font-weight:680;letter-spacing:-.01em}
.ok{color:var(--ok)}.busy{color:var(--busy)}
.pill{display:inline-block;width:.58rem;height:.58rem;border-radius:50%;margin-right:.48rem}
.pill.ok{background:var(--ok);box-shadow:0 0 0 4px color-mix(in srgb,var(--ok) 12%,transparent)}
.pill.busy{background:var(--busy);box-shadow:0 0 0 4px color-mix(in srgb,var(--busy) 12%,transparent);
animation:pulse 1.6s infinite}
@keyframes pulse{50%{opacity:.42;box-shadow:0 0 0 7px transparent}}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:none;color:var(--accent2)}
code{font-family:var(--mono);font-size:.86em;background:color-mix(in srgb,var(--bg) 76%,transparent);
border:1px solid var(--line);padding:.1rem .4rem;border-radius:7px}
pre{font-family:var(--mono);font-size:.79rem;line-height:1.55;
background:color-mix(in srgb,var(--bg) 80%,transparent);border:1px solid var(--line);
padding:.9rem;border-radius:12px;overflow-x:auto;margin:.55rem 0}
details{background:var(--card);border:1px solid var(--line);border-radius:14px;
margin:.55rem 0;padding:.7rem 1rem;box-shadow:var(--shadow-sm);backdrop-filter:blur(16px)}
summary{cursor:pointer;font-weight:680;font-size:.9rem;list-style:none}
summary::-webkit-details-marker{display:none}
details:not(.think)>summary:after{content:"+";float:right;color:var(--muted);font-size:1.15rem;line-height:1}
details[open]:not(.think)>summary:after{content:"−"}
button{font:inherit;font-size:.78rem;font-weight:680;background:var(--card-solid);color:var(--fg);
border:1px solid var(--line-strong);border-radius:10px;padding:.42rem .72rem;cursor:pointer;
transition:transform .15s ease,border-color .15s ease,background .15s ease}
button:hover:not(:disabled){border-color:var(--accent);color:var(--accent);transform:translateY(-1px)}
button:focus-visible,textarea:focus-visible,a:focus-visible,summary:focus-visible{
outline:3px solid color-mix(in srgb,var(--accent) 32%,transparent);outline-offset:2px}
button:disabled{opacity:.42;cursor:not-allowed}
button.primary{background:linear-gradient(135deg,var(--accent),var(--accent2));
border-color:transparent;color:#fff;box-shadow:0 7px 18px color-mix(in srgb,var(--accent) 22%,transparent)}
button.primary:hover:not(:disabled){filter:brightness(1.08);color:#fff}
.endpoint-card{display:flex;align-items:center;justify-content:space-between;gap:1rem;
background:linear-gradient(120deg,var(--card),color-mix(in srgb,var(--accent) 7%,var(--card)))}
.endpoint-card .v code{display:inline-block;max-width:100%;overflow:hidden;text-overflow:ellipsis;
vertical-align:bottom;white-space:nowrap}
.endpoint-actions{text-align:right;white-space:nowrap}
.action-link{display:inline-flex;align-items:center;gap:.25rem;font-size:.82rem;font-weight:700;
padding:.38rem .6rem;border-radius:9px;background:color-mix(in srgb,var(--accent) 9%,transparent)}
canvas.chart{width:100%;height:100px;display:block}
.chartv{font-family:var(--mono);font-size:.96rem;font-weight:750;float:right;color:var(--fg);
letter-spacing:-.04em}
.legend{font-size:.68rem;color:var(--muted)}
.legend i{display:inline-block;width:.9em;height:3px;border-radius:2px;
vertical-align:middle;margin:0 .3em 0 .8em}
.wrap.wide{max-width:70rem}
aside.chatpane{display:none}
@media(min-width:75rem){
 .wrap.wide{max-width:100rem}
 .wrap.wide .layout{grid-template-columns:minmax(0,1fr) 27rem;align-items:start}
 .wrap.wide aside.chatpane{display:flex;flex-direction:column;position:sticky;top:1rem;
  height:calc(100vh - 2rem);background:var(--card);border:1px solid var(--line);
  border-radius:20px;box-shadow:var(--shadow);backdrop-filter:blur(20px);overflow:hidden}
}
.chathead{display:flex;align-items:center;justify-content:space-between;gap:.8rem;
padding:1rem 1rem .8rem;border-bottom:1px solid var(--line)}
.chathead-main{display:flex;align-items:center;gap:.65rem;min-width:0}
.assistant-avatar{display:grid;place-items:center;width:2.15rem;height:2.15rem;border-radius:10px;
background:linear-gradient(145deg,var(--accent),var(--accent2));color:#fff;font-size:.95rem;font-weight:800}
.chathead strong{display:block;font-size:.9rem;line-height:1.15}
.chathead small{display:flex;align-items:center;gap:.35rem;color:var(--muted);font-size:.69rem;margin-top:.2rem}
.chathead small:before{content:"";display:block;width:.43rem;height:.43rem;border-radius:50%;
background:var(--ok);box-shadow:0 0 0 3px color-mix(in srgb,var(--ok) 12%,transparent)}
.expand-link{display:grid;place-items:center;width:2rem;height:2rem;border:1px solid var(--line);
border-radius:9px;color:var(--muted)}
.chatpane #log{flex:1;overflow-y:auto;min-height:0;padding:.8rem}
#log{scrollbar-width:thin;scrollbar-color:var(--line-strong) transparent}
.chatempty{height:100%;min-height:12rem;display:grid;place-content:center;text-align:center;
padding:1.5rem;color:var(--muted)}
.chatempty-icon{display:grid;place-items:center;width:3rem;height:3rem;margin:0 auto .7rem;
border-radius:15px;background:linear-gradient(145deg,color-mix(in srgb,var(--accent) 14%,transparent),
color-mix(in srgb,var(--accent2) 14%,transparent));color:var(--accent);font-size:1.25rem}
.chatempty strong{display:block;color:var(--fg);font-size:.9rem;margin-bottom:.2rem}
.chatempty span{font-size:.76rem;max-width:15rem}
.msg{position:relative;max-width:88%;padding:.65rem .82rem;border-radius:15px;margin:.42rem 0;
white-space:pre-wrap;overflow-wrap:anywhere;font-size:.87rem;width:fit-content;
box-shadow:0 3px 10px rgba(0,0,0,.05)}
.msg.you{background:linear-gradient(130deg,var(--accent),var(--accent2));color:#fff;
margin-left:auto;border-bottom-right-radius:5px}
.msg.bot{background:color-mix(in srgb,var(--card-solid) 72%,transparent);
border:1px solid var(--line);border-bottom-left-radius:5px}
.msg.bot:empty:after{content:"Thinking…";color:var(--muted);animation:softpulse 1.2s infinite}
@keyframes softpulse{50%{opacity:.45}}
details.think{color:var(--muted);font-size:.77em;margin:.3rem 0;max-width:88%;
background:none;border:none;box-shadow:none;padding:0 .3rem;backdrop-filter:none}
details.think summary{color:var(--accent);font-size:.78rem}
details.think>div{margin-top:.35rem;padding:.6rem .7rem;border-left:2px solid var(--line-strong);
white-space:pre-wrap;overflow-wrap:anywhere}
.composer{padding:.72rem;border-top:1px solid var(--line);
background:color-mix(in srgb,var(--card-solid) 55%,transparent)}
.inputrow{display:flex;gap:.5rem;align-items:flex-end}
#in{flex:1;font:inherit;font-size:.86rem;background:color-mix(in srgb,var(--bg) 72%,transparent);
color:var(--fg);border:1px solid var(--line-strong);border-radius:13px;padding:.68rem .75rem;
min-height:2.75rem;max-height:9rem;resize:none}
#in::placeholder{color:var(--faint)}
.composerbar{display:flex;align-items:center;justify-content:space-between;gap:.6rem;margin-top:.52rem}
.chatprefs{display:flex;align-items:center;gap:.75rem;flex-wrap:wrap}
.composer-actions{display:flex;gap:.35rem}
.turncount{align-self:center;color:var(--faint);font-size:.68rem;margin-right:.15rem}
.thinkbox{position:relative;display:flex;align-items:center;gap:.42rem;color:var(--muted);
font-size:.73rem;white-space:nowrap;cursor:pointer;user-select:none}
.thinkbox input{position:absolute;opacity:0;pointer-events:none}
.switch{position:relative;width:1.85rem;height:1.05rem;border-radius:99px;background:var(--line-strong);
transition:background .18s ease}
.switch:after{content:"";position:absolute;top:.15rem;left:.15rem;width:.75rem;height:.75rem;
border-radius:50%;background:var(--card-solid);box-shadow:0 1px 4px rgba(0,0,0,.18);transition:transform .18s ease}
.thinkbox input:checked+.switch{background:var(--accent)}
.thinkbox input:checked+.switch:after{transform:translateX(.8rem)}
.thinkbox input:focus-visible+.switch{outline:3px solid color-mix(in srgb,var(--accent) 30%,transparent)}
@media(max-width:38rem){
 .wrap{padding:.7rem .7rem 2rem}
 header.hero{border-radius:16px;padding:.9rem}.brandmark{width:2.6rem;height:2.6rem;flex-basis:2.6rem}
 .hero-state{display:none}.grid{grid-template-columns:1fr 1fr;gap:.55rem}
 .card{padding:.85rem}.endpoint-card{align-items:flex-start;flex-direction:column}
 .endpoint-actions{text-align:left}.composerbar{align-items:center;flex-wrap:wrap}
 .chatprefs,.composer-actions{width:100%}.composer-actions{justify-content:flex-end}
}
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
var EP="$ep", KEY="$key";
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
const EP="$ep", KEY="$key", msgs=[];
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
    // Reasoning-only responses already use their reasoning as content. Adding
    // it again under a reasoning field would duplicate context.
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
      body:JSON.stringify({model:"$model",messages:wireMessages(),stream:true,max_tokens:$output,
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
        // vLLM reasoning parsers do not all use the same delta field:
        // GLM emits reasoning_content while Qwen emits reasoning.
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
    // Some Qwen/vLLM combinations finish with the useful response in the
    // reasoning stream and null content. Never leave an empty assistant turn.
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

CHAT_PAGE = Template("""<!doctype html><html><head><title>$display chat</title>
<meta name=viewport content="width=device-width,initial-scale=1">""" + STYLE + """<style>
.wrap.chatpage{max-width:56rem;display:flex;flex-direction:column;height:100vh;height:100dvh;padding-bottom:1rem}
.chatpage header.hero{flex:0 0 auto}
.chatpage #log{flex:1;overflow-y:auto;padding:1rem;min-height:0;background:var(--card);
border:1px solid var(--line);border-bottom:none;border-radius:20px 20px 0 0;box-shadow:var(--shadow)}
.chatpage .composer{border:1px solid var(--line);border-radius:0 0 20px 20px;box-shadow:var(--shadow)}
.chatpage .msg{max-width:78%}
</style></head><body><div class="wrap chatpage">
<header class=hero><div class=brandmark aria-hidden=true></div>
<div class=hero-copy><div class=eyebrow>Quick chat</div><h1><b>$display</b></h1>
<span class=sub><a href="/?token=$token">&larr; Back to status &amp; dashboard</a></span></div>
<div class=hero-state><span class="pill ok"></span>Engine online</div></header>
""" + CHAT_WIDGET + CHAT_JS + """<script>inp.focus();</script></div></body></html>""")

PAGE_HEAD = Template("""<!doctype html><html><head><title>$display model turnkey</title>
<meta name=viewport content="width=device-width,initial-scale=1">""" + STYLE +
                     """</head><body>""")


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
        return (f"<div class='card status-card'><h3>{html.escape(label)}</h3>"
                f"<div class=v><span class='pill {cls}'></span>"
                f"<span class={cls}>{html.escape(value)}</span></div></div>")

    real0 = st.get("api_key", "")
    chat_ok = bool(secure and TOKEN and real0 and endpoint and serving)
    wrap_cls = "wrap wide" if chat_ok else "wrap"
    display = html.escape(MODEL_DISPLAY_NAME)
    features = html.escape(LANDING_FEATURES)
    parts = [PAGE_HEAD.substitute(display=display),
             f'<div class="{wrap_cls}"><div class=layout><main>',
             "<header class=hero><div class=brandmark aria-hidden=true></div>"
             f"<div class=hero-copy><div class=eyebrow>Turnkey inference appliance</div>"
             f"<h1><b>{display}</b></h1><span class=sub>{features}</span></div>"
             f"<div class=hero-state><span class='pill {'ok' if serving else 'busy'}'></span>"
             f"{'Engine online' if serving else 'Starting up'}</div></header>",
             "<div class=grid>",
             card("Profile", MODEL_PROFILE, True),
             card("Weights", weights, weights == "ready"),
             card("TLS / DNS", st.get("tls", "not configured"),
                  st.get("tls", "").startswith("https")),
             card("Engine", engine, serving),
             card("DRAM KV offload", st.get("offload", "off"),
                  st.get("offload", "off") != "off"),
             "</div>"]
    if endpoint:
        ep = html.escape(endpoint, quote=True)
        real = st.get("api_key", "")  # from the root-only status file
        key = real if (secure and TOKEN and real) else "<paste API key from instance logs>"
        tok_esc = html.escape(tok, quote=True)
        parts.append(f'<div class="card endpoint-card"><div><h3>OpenAI-compatible endpoint</h3>'
                     f'<div class=v><a href="{ep}/v1/models"><code>{ep}/v1</code></a></div>'
                     f'<div class=sub style="margin-top:.4rem">'
                     f'<a href="{ep}/metrics">Prometheus /metrics</a></div></div>'
                     '<div class=endpoint-actions>')
        if not key.startswith("<"):
            parts.append(f'<a class=action-link href="/chat?token={tok_esc}">Quick chat &rarr;</a>')
        parts.append('</div></div>')
        if key.startswith("<"):
            parts.append("<p class=sub>The API key is printed in the instance logs "
                         "(open Pod/instance logs and look for <code>API KEY</code>).</p>")
        if serving:
            parts.append(METRICS_SECTION.substitute(ep=endpoint, key=key))
        parts.append("<h2>Client configs</h2>")
        for name, where, body in SNIPPETS:
            filled = Template(body).substitute(
                ep=endpoint,
                key=key,
                model=MODEL_ID,
                display=MODEL_DISPLAY_NAME,
                context=MODEL_CONTEXT_WINDOW,
                output=MODEL_OUTPUT_LIMIT,
            )
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
    if not serving:
        # keep boot status fresh; once serving, the dashboard polls instead
        parts.append("<script>setTimeout(function(){location.reload()},20000)</script>"
                     "<p class=sub>Auto-refreshing every 20 s while booting.</p>")
    parts.append("</main>")
    if chat_ok:
        parts.append("<aside class=chatpane><div class=chathead>"
                     "<div class=chathead-main><div class=assistant-avatar aria-hidden=true>✦</div>"
                     "<div><strong>Quick chat</strong><small>Multi-turn session</small></div></div>"
                     f'<a class=expand-link href="/chat?token={tok_esc}" title="Open full chat" '
                     'aria-label="Open full chat">↗</a></div>' + CHAT_WIDGET +
                     "</aside>" + Template(CHAT_JS).substitute(
                         ep=endpoint,
                         key=real0,
                         model=MODEL_ID,
                         output=MODEL_OUTPUT_LIMIT,
                     ))
    parts.append("</div></div></body></html>")
    return "".join(parts).encode()


class DualProtocolServer(ThreadingHTTPServer):
    """Peek the first byte of each connection: TLS handshake (0x16) or plain HTTP."""

    def get_request(self):
        sock, addr = self.socket.accept()
        try:
            first = sock.recv(1, socket.MSG_PEEK)
        except OSError:
            first = b""
        ctx = ssl_ctx()
        if ctx and first == b"\x16":
            try:
                sock = ctx.wrap_socket(sock, server_side=True)
            except ssl.SSLError:
                sock.close()
                raise OSError("TLS handshake failed")
        return sock, addr


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep provider logs quiet
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path not in ("/", "/chat"):
            self.send_error(404)
            return
        tok = parse_qs(url.query).get("token", [""])[0]
        if TOKEN and tok != TOKEN:
            self.send_error(403, "bad token (use the dashboard URL printed in the Pod/instance logs)")
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
            body = CHAT_PAGE.substitute(
                ep=st["endpoint"],
                key=key,
                token=html.escape(tok, quote=True),
                display=html.escape(MODEL_DISPLAY_NAME),
                model=MODEL_ID,
                output=MODEL_OUTPUT_LIMIT,
            ).encode()
        else:
            body = render(secure, tok)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    DualProtocolServer(("0.0.0.0", 1111), Handler).serve_forever()
