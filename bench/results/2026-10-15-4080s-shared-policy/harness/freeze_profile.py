from pathlib import Path
import sys,json,hashlib,datetime
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from tools import prepare_unsloth_q2 as prep
profile=root/'strata-unsloth-4080s-headroom-128k.json'
cfg=prep.refreshed_config(json.loads(profile.read_text()))
prep.admit_automatic_launch(cfg['memory_policy']['sample'])
prep.check_ready(cfg)
assert cfg['host']=='127.0.0.1' and cfg['port']==8081 and cfg.get('api_key')
assert cfg['env']['CUDA_MODULE_LOADING']=='EAGER' and cfg['env']['STRATA_TRACE']=='1'
assert prep.flag_value(cfg['args'],'--max-context')=='131072'
prep.atomic_files({profile:prep.json_bytes(cfg)})
out=Path(__file__).resolve().parent
payload=profile.read_bytes()
(out/'effective-profile.json').write_bytes(payload)
safe={k:v for k,v in cfg.items() if k!='api_key'}
safe['api_key_enabled']=True
safe['frozen_at']=datetime.datetime.now().astimezone().isoformat()
safe['profile_sha256']=hashlib.sha256(payload).hexdigest()
(out/'effective-profile.sanitized.json').write_text(json.dumps(safe,indent=2)+'\n')
prep.report_memory(cfg['memory_policy']['sample'],cfg)
print('Saved frozen automatic profile; SHA-256',safe['profile_sha256'])
