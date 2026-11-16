#!/usr/bin/env python3
"""Part W confirm sequencer: screen -> up to 2 arms, A-B-B-A with shared base loads; a block starts only if it fits."""
import json, subprocess, sys, time
plan_w = json.load(open('plan-w.json'))
scr = json.loads(subprocess.check_output([sys.executable, 'analyze_d.py', 'w', 'cells']))['screen']
json.dump(scr, open('w-screen.json', 'w'), indent=1)
adv = sorted([a for a, r in scr.items() if r['advance']], key=lambda a: (-round(max(scr[a]['prefill_ratio'], scr[a]['mixed_long_ratio']), 3), a))[:2]
arms = {l['arm']: l for l in plan_w['loads']}
D4B = arms['D4b']['args']
conf = [dict(task='prefill', cell='C1', set='L30K', rounds=5), dict(task='mixed', cell='C1', rounds=5, reference='TOKEN-REFERENCE.json')]
base = lambda k: dict(tag='W-Bc%d' % k, engine='tf', arm='D4b', args=D4B, tasks=conf, est_minutes=12, halt_on_fail=True)
print(json.dumps(dict(advance=adv)), flush=True)
k = 1
for i, a in enumerate(adv):
    loads = ([base(k)] if i == 0 else []) + [dict(tag='WC-%s-%d' % (a, j), engine='tf', arm=a, args=arms[a]['args'], env=arms[a].get('env', {}), tasks=conf, est_minutes=12) for j in (1, 2)] + [base(k + 1)]
    cost = sum(l['est_minutes'] for l in loads)
    if time.time() + cost * 60 > plan_w['deadline_epoch']:
        print(json.dumps(dict(arm=a, disposition='unrun', reason='budget: whole confirm block does not fit')), flush=True)
        with open('loads.jsonl', 'a') as f: f.write(json.dumps(dict(tag='WC-' + a, disposition='unrun', reason='budget')) + '\n')
        continue
    json.dump(dict(claim_epoch=plan_w['claim_epoch'], deadline_epoch=plan_w['deadline_epoch'], loads=loads), open('plan-wc-%s.json' % a, 'w'), indent=1)
    rc = subprocess.run([sys.executable, 'run_loads_d.py', 'plan-wc-%s.json' % a]).returncode
    print(json.dumps(dict(arm=a, rc=rc, a1='W-Bc%d' % k, a2='W-Bc%d' % (k + 1))), flush=True)
    k += 1
    if rc != 0:
        break
