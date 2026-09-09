import concurrent.futures,json,pathlib,sys,threading,time,uuid
import benchmark_serving as b
r=pathlib.Path(__file__).resolve().parent
arm=sys.argv[1];out=r/arm;out.mkdir(exist_ok=True)
print("OVERLAP_READY",arm,flush=True)
for repeat in range(3):
 nonce=f"{arm}-{repeat}-{uuid.uuid4().hex}"
 short,ns=b.make_prompt("http://127.0.0.1:8000","","local-primary",1024,nonce+"-short")
 long,nl=b.make_prompt("http://127.0.0.1:8000","","local-primary",65536,nonce+"-long")
 first=threading.Event();before=b.get_metrics("http://127.0.0.1:8000","")
 wall=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  dec=pool.submit(b.stream_completion,"http://127.0.0.1:8000","","local-primary",short,ns,2048,temperature=1,seed=1700+repeat,on_first_text=first.set,trace=True,timeout=600)
  if not first.wait(180):raise RuntimeError("decode did not begin")
  pref=pool.submit(b.stream_completion,"http://127.0.0.1:8000","","local-primary",long,nl,1,temperature=1,seed=2700+repeat,trace=True,timeout=600)
  pr,dr=pref.result(),dec.result()
 after=b.get_metrics("http://127.0.0.1:8000","")
 pt=pr.get("timeline",{});dt=dr.get("timeline",{})
 start,end=pt.get("started_at",0),pt.get("first_text_at",0)
 times=dt.get("text_chunk_times",[])
 gaps=[c-a for a,c in zip(times,times[1:]) if a<end and c>start]
 doc={"repeat":repeat,"started_at":wall,"ended_at":time.time(),"ok":pr.get("ok",False) and dr.get("ok",False),"prefill":pr,"decode":dr,"decode_chunks_during_prefill":sum(start<t<end for t in times),"maximum_inter_chunk_gap_during_prefill_s":max(gaps) if gaps else None,"spec_counters_mixed_with_clients":b.spec_summary(before,after),"server_metric_delta_mixed_with_clients":{k:after[k]-before.get(k,0) for k in after if k.endswith(("_total","_count","_sum"))},"scope":"Request-local HTTP inter-chunk timing; includes queue wait and client contention, not GPU-only timing or isolated serving throughput"}
 b.write_result(str(out/f"overlap-{repeat}.json"),doc)
 print("OVERLAP_RESULT",arm,repeat,doc["ok"],pr.get("ttft_ms"),doc["decode_chunks_during_prefill"],doc["maximum_inter_chunk_gap_during_prefill_s"],flush=True)
 if not doc["ok"]:raise SystemExit(1)
