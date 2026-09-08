import gzip,json,pathlib,statistics,sys
r=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else pathlib.Path(__file__).resolve().parent
if (r/'evidence').is_dir():r=r/'evidence'
summaries={}
monitor_path=r/'monitor.jsonl'
monitor_text=monitor_path.read_text() if monitor_path.exists() else gzip.decompress((r/'monitor.jsonl.gz').read_bytes()).decode()
monitor=[json.loads(l) for l in monitor_text.splitlines()]
for ar in sorted(p for p in r.iterdir() if p.is_dir() and (list(p.glob('hotel*.json')) or list(p.glob('overlap-*.json')))):
 rows=[]
 for p in sorted(ar.glob('hotel*.json')):
  d=json.loads(p.read_text());s=d['selected_summary']
  rows.append({'file':p.name,'concurrency':d['selected_concurrency'],'completed':s['completed'],'errors':s['errors'],'truncated':s['truncated'],'gen_tps':s['aggregate_gen_tok_s'],'e2e_tps':s['aggregate_e2e_tok_s'],'ttft_p50_s':s['ttft']['p50'],'ttft_p90_s':s['ttft']['p90'],'request_gen_tps_p50':s['gen_tok_s']['p50']})
 ov=[]
 for p in sorted(ar.glob('overlap-*.json')):
  d=json.loads(p.read_text());ov.append({'file':p.name,'ok':d['ok'],'prefill_ttft_s':d['prefill'].get('ttft_ms',0)/1000,'decode_tpot_ms':d['decode'].get('tpot_ms'),'max_chunk_gap_s':d['maximum_inter_chunk_gap_during_prefill_s'],'decode_chunks_during_prefill':d['decode_chunks_during_prefill']})
 samples=[x for x in monitor if x.get('arm')==ar.name]
 mins=[min((x['gpus'][i]['free_mib'] for x in samples if len(x.get('gpus',[]))==4),default=None) for i in range(4)]
 logs=(ar/'container-private.log').read_text(errors='replace') if (ar/'container-private.log').exists() else ''
 error_terms=['OutOfMemoryError','CUDA error','VERIFICATION FAILED','vLLM exited','_w4a16_route_count_kernel','_pack_topk_routes_post_prefix_kernel','_pack_topk_routes_sort_kernel']
 summaries[ar.name]={'synthetic':rows,'overlap':ov,'steady_free_mib_min':mins,'monitor_samples':len(samples),'http_unhealthy_samples':sum(x.get('health')!=200 for x in samples),'log_counts':{s:logs.count(s) for s in error_terms} if logs else None,'scope':'Global passive samples include real clients; request-local results are synthetic, all capped/truncated results retained. Missing private logs are not evidence of zero failures.'}
(r/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
for arm,d in summaries.items():
 print(arm,'free min',d['steady_free_mib_min'],'unhealthy',d['http_unhealthy_samples'])
 for c in [1,4]:
  rr=[x for x in d['synthetic'] if x['concurrency']==c]
  if rr: print('C'+str(c),'gen_tps',[round(x['gen_tps'],2) for x in rr],'ttft_p50',[round(x['ttft_p50_s'],3) for x in rr])
 print('overlap',d['overlap'])
