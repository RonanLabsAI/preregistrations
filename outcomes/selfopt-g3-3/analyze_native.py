import sqlite3,json,pathlib,statistics,math,sys
P=pathlib.Path(sys.argv[1]);served=json.loads((P/'served-durations.json').read_text());out=[]
for rows in [1,16]:
 f=P/f'validation-native-c{rows}-r96.sqlite';db=sqlite3.connect(f);q='select s.value,(k.end-k.start)/1000.0 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start';ks=[x for x in db.execute(q) if x[0].startswith('tf_exl3_')];assert len(ks)==720
 stages=[]
 for j in range(6):
  samples=ks[j::6];v=[statistics.median(x[1] for x in samples[i*20:(i+1)*20]) for i in range(1,6)];m=statistics.mean(v);h=2.776445105*statistics.stdev(v)/5**.5;ref=statistics.median(r['kernels'][j]['median_us'] for r in served[f'c{rows}']);stages.append(dict(stage=j,name=samples[0][0],mean_us=m,ci95_us=[m-h,m+h],block_medians_us=v,served_us=ref,ratio=m/ref,pass_tolerance=.85<=m/ref<=1.15))
 out.append(dict(rows=rows,kind='preregistered native-launch sensitivity; primary retained',stages=stages,pass_tolerance=all(s['pass_tolerance'] for s in stages)))
(P/'native-validation.json').write_text(json.dumps(out,indent=2)+'\n')
for x in out:print(x['rows'],x['pass_tolerance'],[(s['stage'],round(s['mean_us'],4),round(s['ratio'],4),s['pass_tolerance']) for s in x['stages']])
