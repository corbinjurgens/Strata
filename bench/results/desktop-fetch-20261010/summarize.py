"""Summarize saved runs; no model operations."""
from pathlib import Path
import json, re
root=Path(__file__).resolve().parent
base=json.loads((root/'baseline-1/results.json').read_text())
records=[]
for path in sorted(root.glob('*/results.json')):
    r=json.loads(path.read_text())
    rows=r.get('rows',[])
    log=path.with_name('engine.log')
    text=log.read_text(errors='replace') if log.exists() else ''
    cache=re.findall(r'recent file expert cache ([\d.]+) MiB',text)
    record={'run':path.parent.name,'status':r.get('status'),'diagnostic':r.get('timing_diagnostic'),
      'summary':r.get('summary'),'held_out_summary':r.get('held_out_summary'),
      'transfer_summary':r.get('transfer_summary'),
      'held_out':[(x['case'],x['passed']) for x in r.get('held_out',[])],
      'long_context_passed':r.get('long_context_passed'),'memory':r.get('memory'),
      'exe_sha256':r.get('exe_sha256'),'recent_cache_mib':float(cache[-1]) if cache else None,
      'engine':{k:r.get('engine',{}).get(k) for k in ('expert_slots','arena_mib','max_context','kv','kv_resident')},
      'logical_decode_file_mb':sum(x.get('metric',{}).get('file_mb',0) for x in rows),
      'transfer_logical_decode_file_mb':sum(x.get('metric',{}).get('file_mb',0) for x in r.get('transfer_workload',[])),
      'exact_choices_vs_baseline':[a['response']['choices']==b['response']['choices'] for a,b in zip(base['rows'],rows)],
      'individual_rates':[x['response']['timings']['predicted_per_second'] for x in rows]}
    records.append(record)
(root/'summary.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
for r in records:
    s=r['summary'] or {}
    print(r['run'],r['status'],'decode',round(s.get('tokens_per_second',0),3),'cacheMiB',r['recent_cache_mib'],
          'minRAM',r['memory'].get('min_available_gib') if r['memory'] else None,'held',r['held_out'])
