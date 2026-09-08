import json,pathlib,subprocess,time,urllib.request
r=pathlib.Path(__file__).resolve().parent
current=json.loads((r/'current.json').read_text());ar=r/current['arm']
rows=[];deadline=time.monotonic()+float(__import__('sys').argv[1])
print('SOAK_READY',current['arm'],flush=True)
def metrics():
 with urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=10) as f:
  out={}
  for line in f.read().decode().splitlines():
   if line.startswith('#') or not line.startswith(('vllm:','http_requests_total')):continue
   try:k,v=line.rsplit(' ',1);out[k]=float(v)
   except ValueError:pass
  return out
first=metrics()
while time.monotonic()<deadline:
 try:
  h=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=10).status
 except Exception as e:h=type(e).__name__
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],timeout=15).decode()
 d=json.loads(subprocess.check_output(['podman','inspect',current['name']],timeout=20))[0]
 rows.append({'time':time.time(),'health':h,'free_mib':[float(l.split(',')[1]) for l in gpu.splitlines()],'restarts':d['RestartCount'],'status':d['State']['Status'],'container_health':d['State'].get('Health',{}).get('Status'),'oom_killed':d['State'].get('OOMKilled')})
 time.sleep(30)
last=metrics()
delta={k:last[k]-first.get(k,0.0) for k in last if any(t in k for t in ['request_success_total','num_preemptions_total','http_requests_total','spec_decode'])}
doc={'arm':current['arm'],'container':current['name'],'samples':len(rows),'unhealthy_samples':sum(x['health']!=200 for x in rows),'restarts_observed':max(x['restarts'] for x in rows),'oom_killed_observed':any(x['oom_killed'] for x in rows),'min_free_mib_per_gpu':[min(x['free_mib'][i] for x in rows) for i in range(4)],'counter_delta_includes_real_clients':{k:round(v,3) for k,v in delta.items() if v},'scope':'Global counters include real client traffic; not an isolated benchmark.'}
(ar/'soak.json').write_text(json.dumps(doc,indent=2)+'\n');(ar/'soak-samples.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
print(json.dumps(doc),flush=True);print('SOAK_COMPLETE',flush=True)
