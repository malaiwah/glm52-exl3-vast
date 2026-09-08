import pathlib,subprocess,sys
r=pathlib.Path(__file__).resolve().parent;arm=sys.argv[1]
print('FAIRNESS_MEASURE_READY',arm,flush=True)
subprocess.run(['python3','-u',str(r/'overlap.py'),arm],check=True)
subprocess.run(['python3',str(r/'benchmark_serving.py'),'--base-url','http://127.0.0.1:8000','--model','local-primary','--concurrency','8,12','--requests-per-level','12','--input-tokens','1024','--output-tokens','128','--prefill-tokens','','--warmup','0','--temperature','1','--seed','810','--timeout','600','--out',str(r/arm/'concurrency-gate.json')],check=True)
print('FAIRNESS_MEASURE_COMPLETE',arm,flush=True)
