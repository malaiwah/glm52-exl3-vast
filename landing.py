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
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/GLM-5.2-EXL3-TR3-3.0bpw")
STATUS_FILE = os.environ.get("STATUS_FILE", "/tmp/glm-boot-status.json")
ALLOW_INSECURE = os.environ.get("LANDING_ALLOW_INSECURE", "0") == "1"
API_PORT = os.environ.get("LANDING_API_PORT", "8000")
WEIGHTS_TOTAL_GIB = 309  # 332 GB

_ssl_ctx = None


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
    return f"downloading — {done / 2**30:.0f} of ~{WEIGHTS_TOTAL_GIB} GiB"


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
    "glm-vast": {
      "baseUrl": "$ep/v1",
      "api": "openai-completions",
      "apiKey": "$key",
      "models": [
        {"id": "GLM-5.2", "name": "GLM-5.2 (vast)", "contextWindow": 524288}
      ]
    }
  }
}"""),
    ("opencode", "opencode.json (project or ~/.config/opencode/)", """{
  "$$schema": "https://opencode.ai/config.json",
  "provider": {
    "glm-vast": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "GLM-5.2 (vast)",
      "options": {"baseURL": "$ep/v1", "apiKey": "$key"},
      "models": {
        "GLM-5.2": {"name": "GLM-5.2", "limit": {"context": 524288, "output": 131072}}
      }
    }
  }
}"""),
    ("Claude Code", "shell (native /v1/messages — Anthropic wire format)", """export ANTHROPIC_BASE_URL="$ep"
export ANTHROPIC_AUTH_TOKEN="$key"
export ANTHROPIC_MODEL="GLM-5.2"
claude"""),
    ("Codex", "~/.codex/config.toml", """model = "GLM-5.2"
model_provider = "glm-vast"

[model_providers.glm-vast]
name = "GLM-5.2 (vast)"
base_url = "$ep/v1"
env_key = "GLM_API_KEY"
wire_api = "chat"

# then:  export GLM_API_KEY="$key" """),
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


CHAT_WIDGET = """<div id=log></div>
<div class=composer>
<textarea id=in placeholder="Message (Ctrl+Enter to send)"></textarea>
<div class=side>
<button id=send class=primary>Send</button>
<button id=stop disabled>Stop</button>
<button id=clear>Clear</button>
<label class=thinkbox><input type=checkbox id=think checked> thinking</label>
</div></div>"""

CHAT_JS = """<script>
const EP="$ep", KEY="$key", msgs=[];
const log=document.getElementById("log"), inp=document.getElementById("in");
let ctrl=null;
function el(tag,cls,txt){const e=document.createElement(tag);if(cls)e.className=cls;if(txt)e.textContent=txt;log.appendChild(e);log.scrollTop=log.scrollHeight;return e}
async function send(){
  const text=inp.value.trim(); if(!text||ctrl)return;
  inp.value=""; msgs.push({role:"user",content:text}); el("div","msg you",text);
  const think=el("details","think"); think.appendChild(document.createElement("summary")).textContent="thinking…";
  const tbody=think.appendChild(document.createElement("div"));
  const out=el("div","msg bot","");
  ctrl=new AbortController(); document.getElementById("stop").disabled=false;
  let answer="", reasoning="";
  try{
    const r=await fetch(EP+"/v1/chat/completions",{method:"POST",signal:ctrl.signal,
      headers:{"Authorization":"Bearer "+KEY,"Content-Type":"application/json"},
      body:JSON.stringify({model:"GLM-5.2",messages:msgs,stream:true,max_tokens:8192,
        chat_template_kwargs:{enable_thinking:document.getElementById("think").checked}})});
    if(!r.ok){out.textContent="HTTP "+r.status+": "+await r.text();return}
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf="";
    for(;;){const {done,value}=await rd.read(); if(done)break;
      buf+=dec.decode(value,{stream:true}); const lines=buf.split("\\n"); buf=lines.pop();
      for(const ln of lines){ if(!ln.startsWith("data: ")||ln.includes("[DONE]"))continue;
        const ch=JSON.parse(ln.slice(6)).choices;
        if(!ch||!ch.length)continue; // final usage chunk has choices:[]
        const d=ch[0].delta||{};
        if(d.reasoning_content){reasoning+=d.reasoning_content;tbody.textContent=reasoning;think.open=true}
        if(d.content){answer+=d.content;out.textContent=answer;
          if(think.open){think.open=false;think.querySelector("summary").textContent="thinking ("+reasoning.length+" chars)"}}
        log.scrollTop=log.scrollHeight;}}
  }catch(e){if(e.name!=="AbortError")out.textContent+="\\n[error: "+e+"]"}
  finally{ctrl=null;document.getElementById("stop").disabled=true;
    if(!reasoning)think.remove();
    msgs.push({role:"assistant",content:answer});}
}
document.getElementById("send").onclick=send;
document.getElementById("stop").onclick=()=>ctrl&&ctrl.abort();
document.getElementById("clear").onclick=()=>{msgs.length=0;log.innerHTML=""};
inp.addEventListener("keydown",e=>{if(e.ctrlKey&&e.key==="Enter")send()});
</script>"""

CHAT_PAGE = Template("""<!doctype html><html><head><title>GLM-5.2 chat</title>
<meta name=viewport content="width=device-width,initial-scale=1">""" + STYLE + """<style>
.wrap{max-width:52rem;display:flex;flex-direction:column;height:100vh;padding-bottom:1rem}
#log{flex:1;overflow-y:auto;padding:.5rem 0}
.msg.bot{background:var(--card)}
</style></head><body><div class=wrap>
<header class=hero><h1><b>GLM-5.2</b> quick chat</h1>
<span class=sub><a href="/?token=$token">&larr; status &amp; dashboard</a></span></header>
""" + CHAT_WIDGET + CHAT_JS + """<script>inp.focus();</script></div></body></html>""")

PAGE_HEAD = ("""<!doctype html><html><head><title>GLM-5.2 EXL3 turnkey</title>
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
        return (f"<div class=card><h3>{html.escape(label)}</h3>"
                f"<div class=v><span class='pill {cls}'></span>"
                f"<span class={cls}>{html.escape(value)}</span></div></div>")

    real0 = st.get("api_key", "")
    chat_ok = bool(secure and TOKEN and real0 and endpoint and serving)
    wrap_cls = "wrap wide" if chat_ok else "wrap"
    parts = [PAGE_HEAD,
             f'<div class="{wrap_cls}"><div class=layout><main>',
             "<header class=hero><h1><b>GLM-5.2</b> EXL3 turnkey</h1>"
             "<span class=sub>512K context &middot; fp8 KV &middot; MTP-3 &middot; "
             "4&times; RTX PRO 6000</span></header>",
             "<div class=grid>",
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
        parts.append(f'<div class=card><h3>OpenAI-compatible endpoint</h3>'
                     f'<div class=v><a href="{ep}/v1/models"><code>{ep}/v1</code></a></div>'
                     f'<div class=sub style="margin-top:.4rem">'
                     f'<a href="{ep}/metrics">Prometheus /metrics</a>')
        if not key.startswith("<"):
            parts.append(f' &middot; <a href="/chat?token={tok_esc}"><b>Quick chat &rarr;</b></a>')
        parts.append('</div></div>')
        if key.startswith("<"):
            parts.append("<p class=sub>The API key is printed in the instance logs "
                         "(vast console &rarr; Logs, look for <code>API KEY</code>).</p>")
        if serving:
            parts.append(METRICS_SECTION.substitute(ep=endpoint, key=key))
        parts.append("<h2>Client configs</h2>")
        for name, where, body in SNIPPETS:
            filled = Template(body).substitute(ep=endpoint, key=key)
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
    if not serving:
        # keep boot status fresh; once serving, the dashboard polls instead
        parts.append("<script>setTimeout(function(){location.reload()},20000)</script>"
                     "<p class=sub>Auto-refreshing every 20 s while booting.</p>")
    parts.append("</main>")
    if chat_ok:
        parts.append("<aside class=chatpane><h2>Quick chat</h2>" + CHAT_WIDGET +
                     "</aside>" + Template(CHAT_JS).substitute(ep=endpoint, key=real0))
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
    def log_message(self, *args):  # keep the vast console log quiet
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path not in ("/", "/chat"):
            self.send_error(404)
            return
        tok = parse_qs(url.query).get("token", [""])[0]
        if TOKEN and tok != TOKEN:
            self.send_error(403, "bad token (use the vast console Open button)")
            return
        secure = isinstance(self.connection, ssl.SSLSocket) or ALLOW_INSECURE
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
            body = CHAT_PAGE.substitute(ep=st["endpoint"], key=key,
                                        token=html.escape(tok, quote=True)).encode()
        else:
            body = render(secure, tok)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    DualProtocolServer(("0.0.0.0", 1111), Handler).serve_forever()
