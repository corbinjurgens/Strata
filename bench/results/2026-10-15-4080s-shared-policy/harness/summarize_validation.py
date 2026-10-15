from pathlib import Path
import json,hashlib,datetime,subprocess
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
GIB=2**30

def phase(name):
    folder=OUT/name
    result=json.loads((folder/'results.json').read_text())
    telemetry=json.loads((folder/'telemetry-summary.json').read_text())
    rows=[json.loads(line) for line in (folder/'telemetry.jsonl').read_text().splitlines() if line.strip()]
    memory=[row for row in rows if row.get('kind')=='memory']
    requests=[]
    for item in result['requests']:
        response=item.get('response',{})
        usage=response.get('usage',{})
        timing=response.get('timings',{})
        requests.append({'label':item['label'],'prompt_tokens':usage.get('prompt_tokens'),'completion_tokens':usage.get('completion_tokens'),'cached_tokens':timing.get('cache_n'),'new_prompt_tokens':timing.get('prompt_n'),'prefill_seconds':timing.get('prompt_ms',0)/1000,'decode_tokens_per_second':timing.get('predicted_per_second'),'wall_seconds':item.get('finished_unix',0)-item['started_unix'] if item.get('finished_unix') else None,'error':item.get('error')})
    return {'status':result['status'],'error':result.get('error'),'smoke_passed':result.get('smoke_passed'),'watched_short_passed':result.get('watched_short_passed'),'long_context_passed':result.get('long_context_passed'),'long_context_checks':[{'case':row['case'],'checks':row['checks'],'passed':row['passed']} for row in result.get('long_context',[])],'watched_checks':[{'label':row['label'],'checks':row['checks'],'passed':row['passed']} for row in result.get('watched_short',[])],'memory':{'minimum_available_physical_gib':min(row['physical_available_bytes'] for row in memory)/GIB,'maximum_used_physical_gib':max(row['physical_used_bytes'] for row in memory)/GIB,'minimum_available_commit_gib':min(row['commit_available_bytes'] for row in memory)/GIB,'maximum_system_commit_gib':max(row['commit_total_bytes'] for row in memory)/GIB,'minimum_commit_limit_gib':min(row['commit_limit_bytes'] for row in memory)/GIB,'maximum_commit_limit_gib':max(row['commit_limit_bytes'] for row in memory)/GIB,'peak_owned_rss_gib':max(row['owned_rss_bytes'] for row in memory)/GIB,'peak_owned_private_gib':max(row['owned_private_bytes'] for row in memory)/GIB,'first_sample':memory[0],'last_sample':memory[-1]},'guard':telemetry,'requests':requests}

summary={'recorded_at':datetime.datetime.now().astimezone().isoformat(),'candidate_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'preparation':json.loads((OUT/'preparation.json').read_text()),'hardware':json.loads((OUT/'hardware.json').read_text()),'effective_profile':json.loads((OUT/'effective-profile.sanitized.json').read_text()),'phases':{}}
if (OUT/'post-snapshot.json').exists(): summary['post_cleanup']=json.loads((OUT/'post-snapshot.json').read_text())
for name in ('short-01','long-01'):
    if (OUT/name/'results.json').exists(): summary['phases'][name]=phase(name)
baseline=json.loads((OUT/'baseline.json').read_text())
summary['baseline_changed_files']=[name for name,row in baseline['files'].items() if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest()!=row['sha256']]
summary['profile_still_frozen']=hashlib.sha256((ROOT/'strata-unsloth-4080s-headroom-128k.json').read_bytes()).hexdigest()==summary['effective_profile']['profile_sha256']
summary['desktop_acceptance_passed']=len(summary['phases'])==2 and all(p['status']=='complete' and not p['guard']['stop_reason'] for p in summary['phases'].values()) and summary['phases']['short-01']['watched_short_passed'] is True and summary['phases']['long-01']['long_context_passed'] is True and not summary['baseline_changed_files'] and summary['profile_still_frozen']
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'desktop_acceptance_passed':summary['desktop_acceptance_passed'],'baseline_changed_files':summary['baseline_changed_files'],'profile_still_frozen':summary['profile_still_frozen'],'phases':{name:{'status':p['status'],'memory':{key:value for key,value in p['memory'].items() if not isinstance(value,dict)}} for name,p in summary['phases'].items()}},indent=2))
