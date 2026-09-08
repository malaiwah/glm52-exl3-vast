import json,pathlib,subprocess,sys,time
r=pathlib.Path(__file__).resolve().parent
arm=sys.argv[1]
(r/arm).mkdir(exist_ok=True)
print("BENCH_READY",arm,flush=True)
for repeat in range(3):
 for concurrency in (1,4):
  label=f"hotel-c{concurrency}-r{repeat}"
  cmd=["/mnt/vault/llm/llm-inference-bench/.venv/bin/python",str(r/"llm_decode_bench.py"),"--host","http://127.0.0.1:8000","--model","local-primary","--test-profile","hotel-lights","--profile-runs","8","--profile-concurrency",str(concurrency),"--max-tokens","2048","--completion-stats-temperature","1","--completion-stats-top-p","0.95","--completion-stats-seed",str(410+repeat*8),"--completion-stats-request-timeout","240","--reasoning-effort","low","--display-mode","plain","--no-resume","--output",str(r/arm/(label+".json"))]
  started=time.time()
  with (r/arm/(label+".log")).open("w") as log:
   p=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
  with (r/arm/"bench-events.jsonl").open("a") as f:f.write(json.dumps({"label":label,"start":started,"end":time.time(),"rc":p.returncode,"cmd":cmd,"traffic_scope":"synthetic request-local results; server counters include unrelated live clients"})+"\n")
  print("BENCH_CELL",arm,label,p.returncode,flush=True)
  if p.returncode:raise SystemExit(p.returncode)
  report=json.loads((r/arm/(label+".json")).read_text())
  if report["selected_summary"]["errors"] or report["selected_summary"]["completed"] != 8:
   raise RuntimeError(label+" failed request-completion gate; receipts retained")
print("BENCH_COMPLETE",arm,flush=True)
