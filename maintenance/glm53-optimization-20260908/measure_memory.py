import json,pathlib,subprocess,sys,time
r=pathlib.Path(__file__).resolve().parent;arm=sys.argv[1];ar=r/arm
print('MEMORY_MEASURE_READY',arm,flush=True)
started=time.time()
def snapshot(label):
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,power.draw','--format=csv,noheader,nounits']).decode()
 (ar/(label+'.json')).write_text(json.dumps({'time':time.time(),'gpus':[dict(zip(('index','used_mib','free_mib','watts'),map(float,l.split(',')))) for l in text.splitlines()]},indent=2)+'\n')
snapshot('memory-before-long')
cmd=['python3',str(r/'verify_serving.py'),'--base-url','http://127.0.0.1:8000','--model','local-primary','--max-model-len','520192','--needle-tokens','500000','--needle-timeout','1800','--needle-seed',str(202609080+int(sys.argv[2])),'--out',str(ar/'verify-500k.json')]
subprocess.run(cmd,check=True)
d=json.loads((ar/'verify-500k.json').read_text());print('LONG_GATE_COMPLETE',arm,flush=True)
snapshot('memory-after-long')
subprocess.run(['python3','-u',str(r/'overlap.py'),arm],check=True)
snapshot('memory-after-overlap')
(ar/'memory-window.json').write_text(json.dumps({'start':started,'end':time.time(),'standardized_warmup':'fresh 500K retrieval followed by 3x64K overlap'})+'\n')
print('MEMORY_MEASURE_COMPLETE',arm,flush=True)
