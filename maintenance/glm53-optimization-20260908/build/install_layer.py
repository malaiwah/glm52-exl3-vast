import hashlib,json,pathlib,py_compile,subprocess
root=pathlib.Path('/opt/turnkey-experiment')
manifest=json.loads((root/'build-manifest.json').read_text())
for entry in manifest['files']:
 source=root/entry['source'];data=source.read_bytes()
 assert hashlib.sha256(data).hexdigest()==entry['candidate_sha256'],source
 targets=[pathlib.Path(entry.get('install_target','/opt/venv/lib/python3.12/site-packages/'+entry['path']))]
 if not entry.get('install_target'):
  mirror=pathlib.Path('/opt')/entry['path'].split('/')[0]/entry['path']
  if mirror.parent.exists(): targets.append(mirror)
 for target in targets:
  if entry['baseline_sha256'] is None: assert not target.exists(),target
  else: assert hashlib.sha256(target.read_bytes()).hexdigest()==entry['baseline_sha256'],target
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  if target.suffix=='.py':py_compile.compile(str(target),doraise=True)
  elif target.suffix=='.sh':subprocess.run(['bash','-n',str(target)],check=True)
print('INSTALLED_VERIFIED',len(manifest['files']))
