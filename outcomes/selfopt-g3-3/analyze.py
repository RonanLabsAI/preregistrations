"""Analyze frozen stage medians and paired empirical ratios without pruning."""
import hashlib,json,math,pathlib,sqlite3,statistics,sys
P=pathlib.Path(sys.argv[1]);served=json.loads((P/'served-durations.json').read_text())
def interval(values,critical):
    mean=statistics.mean(values);half=critical*statistics.stdev(values)/len(values)**.5
    return dict(mean=mean,ci95=[mean-half,mean+half],n=len(values),values=values)
def ratio(a,b):
    logs=[math.log(x/y) for x,y in zip(a,b)];s=interval(logs,2.262157163)
    return dict(speedup=math.exp(s['mean']),ci95=[math.exp(x) for x in s['ci95']],champion_us=a,candidate_us=b,n=len(a))
def kernels(path):
    c=sqlite3.connect(path)
    q='select s.value,gridX,gridY,gridZ,(k.end-k.start)/1000.0 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start'
    return [r for r in c.execute(q) if r[0].startswith('tf_exl3_')]
validation=[];candidates=[]
for rows in [1,16]:
    f=P/f'validation-c{rows}-r96.sqlite'
    if not f.exists():continue
    meta=json.loads(f.with_suffix('.json').read_text());ks=kernels(f);assert len(ks)==720,len(ks)
    expected=['tf_exl3_group','tf_exl3_experts_rot_in','tf_exl3_grouped_1','tf_exl3_gateup_epilogue','tf_exl3_grouped_1','tf_exl3_down_combine']
    assert all(k[0]==expected[i%6] for i,k in enumerate(ks))
    stages=[]
    for j in range(6):
        v=ks[j::6];medians=[statistics.median(r[4] for r in v[i*20:(i+1)*20]) for i in range(1,6)]
        s=interval(medians,2.776445105);ref=statistics.median(x['kernels'][j]['median_us'] for x in served[f'c{rows}'])
        stages.append(dict(stage=j,name=v[0][0],served_us=ref,ratio=s['mean']/ref,pass_tolerance=.85<=s['mean']/ref<=1.15,us=s))
    validation.append(dict(rows=rows,unique=meta['unique'],stages=stages,pass_tolerance=all(x['pass_tolerance'] for x in stages),captured_bitwise=meta['champion_captured_bitwise']))
for f in sorted(P.glob('ab-c*-*.json')):
    if f.name.endswith('-guard.json'):continue
    meta=json.loads(f.read_text());rows=meta['rows'];candidate=meta['candidate'];ks=kernels(f.with_suffix('.sqlite'));assert len(ks)==2640,(f.name,len(ks))
    stage_arms={arm:[[] for _ in range(6)] for arm in ['champion',candidate]}
    for i,block in enumerate(meta['blocks']):
        if block['repeat']<0:continue
        for a,arm in enumerate(block['order']):
            for j in range(6):
                v=[ks[(i*40+a*20+r)*6+j][4] for r in range(20)]
                stage_arms[arm][j].append(statistics.median(v))
    per_stage=[ratio(stage_arms['champion'][j],stage_arms[candidate][j]) for j in range(6)]
    event={arm:[statistics.median(b['us'][arm]) for b in meta['blocks'] if b['repeat']>=0] for arm in stage_arms}
    full=ratio(event['champion'],event[candidate]);share={1:.360485,16:.590868}[rows]
    def bound(r):return 1/(1-share+share/r)
    bounds=dict(point=bound(full['speedup']),ci95=[bound(x) for x in full['ci95']],share=share,applicable=any(v['rows']==rows and v['pass_tolerance'] for v in validation),conditional='unchanged non-EXL3 work; single captured layer/sequence')
    candidates.append(dict(rows=rows,candidate=candidate,stage_speedups=per_stage,full_sequence=full,bounds=bounds,numerics=meta['numerics'],resources=meta['resources'],allocated_bytes=meta['allocated_bytes']))
result=dict(validation=validation,candidates=candidates)
(P/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
