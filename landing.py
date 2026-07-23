#!/usr/bin/env python3
"""vast.ai "Open" button landing page — dual-protocol (TLS + plain) on :1111.

Started first thing in entrypoint.sh so status is visible from the very first
boot minute. Each connection's first byte is peeked: 0x16 = TLS handshake,
else plain HTTP. Once a cert exists (lazy-loaded mid-boot), plain hits are
302-redirected to the HTTPS view. Boot state comes from the status file the
entrypoint rewrites at each milestone, plus live probes (download size,
/health). The API key and filled-in client configs (oh-my-pi, opencode,
Claude Code, Codex) render only over TLS with a validated OPEN_BUTTON_TOKEN.
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
    # vLLM may serve TLS or plain on :8000 — probe both; any failure moves on
    for scheme, kw in (("https", {"context": ssl._create_unverified_context()}),
                       ("http", {})):
        try:
            with urllib.request.urlopen(f"{scheme}://localhost:8000/health",
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

CHAT_PAGE = Template("""<!doctype html><html><head><title>GLM-5.2 chat</title>
<style>body{font-family:system-ui;max-width:52rem;margin:1.5rem auto;padding:0 1rem;line-height:1.45}
#log{border:1px solid #ccc;border-radius:8px;padding:1rem;min-height:50vh;max-height:65vh;overflow-y:auto}
.u{color:#046;font-weight:600;margin-top:.8rem}.a{white-space:pre-wrap}
details.think{color:#888;font-size:.85em;margin:.3rem 0}
#in{width:100%;box-sizing:border-box;min-height:4rem;margin-top:.6rem;font:inherit}
button{font:inherit;padding:.35rem 1rem;margin-right:.5rem}
@media(prefers-color-scheme:dark){body{background:#111;color:#ddd}#log{border-color:#444}.u{color:#8cf}}
</style></head><body>
<h2>GLM-5.2 &mdash; quick chat <small><a href="/?token=$token">&larr; status</a></small></h2>
<div id=log></div>
<textarea id=in placeholder="Message (Ctrl+Enter to send)"></textarea>
<p><button id=send>Send</button><button id=stop disabled>Stop</button>
<button id=clear>Clear</button>
<label><input type=checkbox id=think checked> thinking</label></p>
<script>
const EP="$ep", KEY="$key", msgs=[];
const log=document.getElementById("log"), inp=document.getElementById("in");
let ctrl=null;
function el(tag,cls,txt){const e=document.createElement(tag);if(cls)e.className=cls;if(txt)e.textContent=txt;log.appendChild(e);log.scrollTop=log.scrollHeight;return e}
async function send(){
  const text=inp.value.trim(); if(!text||ctrl)return;
  inp.value=""; msgs.push({role:"user",content:text}); el("div","u","you"); el("div","a",text);
  el("div","u","GLM-5.2");
  const think=el("details","think"); think.appendChild(document.createElement("summary")).textContent="thinking…";
  const tbody=think.appendChild(document.createElement("div")); tbody.className="a";
  const out=el("div","a","");
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
        const d=JSON.parse(ln.slice(6)).choices[0].delta||{};
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
</script></body></html>""")

PAGE_HEAD = """<!doctype html><html><head><title>GLM-5.2 EXL3 turnkey</title>
<meta http-equiv="refresh" content="30">
<style>body{font-family:system-ui;max-width:46rem;margin:3rem auto;line-height:1.5;padding:0 1rem}
code{background:#eee;padding:.1rem .3rem;border-radius:4px}
pre{background:#f4f4f4;padding:.8rem;border-radius:6px;overflow-x:auto}
table{border-collapse:collapse}td{padding:.2rem .8rem .2rem 0}
.ok{color:#2a2}.busy{color:#b80}
details{margin:.6rem 0}summary{cursor:pointer;font-weight:600}
@media(prefers-color-scheme:dark){body{background:#111;color:#ddd}
code,pre{background:#222}}</style></head><body>"""


def render(secure: bool, tok: str = "") -> bytes:
    st = status()
    endpoint = st.get("endpoint", "")
    weights = weights_state()
    engine = engine_state()

    def row(label, value, ok):
        cls = "ok" if ok else "busy"
        return (f"<tr><td>{html.escape(label)}</td>"
                f"<td class={cls}>{html.escape(value)}</td></tr>")

    if engine == "serving":
        weights = "ready"  # a serving engine is proof enough (pre-marker volumes)
    parts = [PAGE_HEAD, "<h1>GLM-5.2 EXL3 turnkey</h1><table>",
             row("Weights", weights, weights == "ready"),
             row("TLS / DNS", st.get("tls", "not configured"),
                 st.get("tls", "").startswith("https")),
             row("Engine", engine, engine == "serving"),
             row("DRAM KV offload", st.get("offload", "off"),
                 st.get("offload", "off") != "off"),
             "</table>"]
    if endpoint:
        ep = html.escape(endpoint, quote=True)
        # Real key only over TLS AND behind an active token gate.
        real = st.get("api_key", "")  # from the root-only status file
        key = real if (secure and TOKEN and real) else "<paste API key from instance logs>"
        parts.append(f'<p>OpenAI-compatible API, 524,288-token context: '
                     f'<a href="{ep}/v1/models"><code>{ep}/v1</code></a> &middot; '
                     f'<a href="{ep}/metrics">Prometheus metrics</a></p>')
        if key.startswith("<"):
            parts.append("<p>The API key is printed in the instance logs "
                         "(vast console &rarr; Logs, look for <code>API KEY</code>).</p>")
        if not key.startswith("<"):
            parts.append(f'<p><a href="/chat?token={html.escape(tok, quote=True)}"><b>Quick chat</b></a> — minimal '
                         'multi-turn test UI (streams straight to the endpoint).</p>')
        parts.append("<h2>Client configs</h2>")
        for name, where, body in SNIPPETS:
            filled = Template(body).substitute(ep=endpoint, key=key)
            parts.append(f"<details><summary>{html.escape(name)}</summary>"
                         f"<p><code>{html.escape(where)}</code></p>"
                         f"<pre>{html.escape(filled)}</pre></details>")
        parts.append(
            '<h2>Quick test <button id=copybtn onclick="copyQT()" '
            'style="font-size:.55em;vertical-align:middle;cursor:pointer">copy</button></h2>'
            f"<pre id=qt>curl -H \"Authorization: Bearer {html.escape(key)}\" {ep}/v1/models</pre>"
            "<script>function copyQT(){navigator.clipboard.writeText("
            "document.getElementById('qt').textContent).then(()=>{"
            "const b=document.getElementById('copybtn');b.textContent='copied!';"
            "setTimeout(()=>{b.textContent='copy'},1500)})}</script>")
    parts.append("<p><small>Page auto-refreshes every 30 s.</small></p></body></html>")
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
        secure = isinstance(self.connection, ssl.SSLSocket)
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
            # the chat page embeds the key in JS: TLS + active token gate required
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
