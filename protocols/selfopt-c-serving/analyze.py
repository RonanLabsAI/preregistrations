#!/usr/bin/env python3
"""Study C analysis: cell summaries, phase-0 bands and A/R ratios, screen ratios, phase-2 paired t95.

  analyze.py summary CELL.jsonl...                 per cell/mode: retained values, mean, CV, t95
  analyze.py phase0 C1.jsonl C4.jsonl C16.jsonl     gates, frozen drafts-on bands, A/R paired ratios
  analyze.py screen PHASE0.json ARM CELL.jsonl...  screen ratios against phase-0 A
  analyze.py confirm CELL A1.jsonl B1.jsonl B2.jsonl A2.jsonl BANDS.json   paired log-ratio t95 df9
"""
import json, math, pathlib, statistics, sys

T95 = {2: 4.302652730, 4: 2.776445105, 9: 2.262157163}
OLD_BANDS = {'C1': (80.12335390502187, 83.39369488073706), 'C16': (267.3277037430555, 278.23903858971084)}


def rounds(path, mode='drafted'):
    rows = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if 'metric' in r and r['mode'] == mode]
    return rows


def retained(path, mode='drafted'):
    rs = rounds(path, mode)
    assert all(r['ids_pass'] for r in rs), 'ID gate failed in %s' % path
    return [r['metric'] for r in rs if not r['warmup']]


def describe(vals):
    n = len(vals)
    m = statistics.fmean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    d = dict(n=n, values=vals, mean=m, sd=sd, cv=sd / m if m else None)
    if n - 1 in T95:
        h = T95[n - 1] * sd / math.sqrt(n)
        d['t95'] = [m - h, m + h]
    return d


def paired(a, b):
    """Mean of log(b/a) over pairs with Student t95; endpoints back-transformed."""
    assert len(a) == len(b) and len(a) - 1 in T95
    logs = [math.log(y / x) for x, y in zip(a, b)]
    n = len(logs)
    m = statistics.fmean(logs)
    sd = statistics.stdev(logs)
    h = T95[n - 1] * sd / math.sqrt(n)
    return dict(n=n, df=n - 1, log_ratios=logs, ratio=math.exp(m), t95=[math.exp(m - h), math.exp(m + h)])


def phase0(paths):
    out = dict(cells={}, baseline_pass=True)
    for p in paths:
        cell = rounds(p, 'drafted')[0]['cell']
        allr = [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]
        ids_ok = {m: all(r['ids_pass'] for r in allr if r.get('mode') == m and 'ids_pass' in r) for m in ('drafted', 'plain')}
        c = dict(ids_pass=ids_ok)
        if ids_ok['drafted']:
            a = describe(retained(p, 'drafted'))
            c['A'] = a
            c['A_cv_pass'] = a['cv'] <= 0.015
            c['band_drafts_on'] = [a['mean'] * 0.98, a['mean'] * 1.02]
        if ids_ok['plain']:
            r = describe(retained(p, 'plain'))
            c['R'] = r
            c['R_cv_le_1p5'] = r['cv'] <= 0.015
            if cell in OLD_BANDS:
                lo, hi = OLD_BANDS[cell]
                c['R_in_original_band'] = lo <= r['mean'] <= hi
        if ids_ok['drafted'] and ids_ok['plain']:
            c['A_over_R'] = paired(c['R']['values'], c['A']['values'])
        c['baseline_pass'] = ids_ok['drafted'] and c.get('A_cv_pass', False)
        c['reference_ids_pass'] = ids_ok['plain']
        out['cells'][cell] = c
        out['baseline_pass'] &= c['baseline_pass'] and c['reference_ids_pass']
    return out


def screen(p0, arm, paths):
    out = dict(arm=arm, cells={})
    for p in paths:
        rs = rounds(p, 'drafted')
        if not rs:
            continue
        cell = rs[0]['cell']
        ok = all(r['ids_pass'] for r in rs)
        c = dict(ids_pass=ok)
        if ok:
            d = describe([r['metric'] for r in rs if not r['warmup']])
            c.update(B=d, ratio=d['mean'] / p0['cells'][cell]['A']['mean'])
        out['cells'][cell] = c
    out['ids_pass'] = all(c['ids_pass'] for c in out['cells'].values())
    rat = [c['ratio'] for c in out['cells'].values() if 'ratio' in c]
    out['best_ratio'] = max(rat) if rat else None
    out['advance'] = out['ids_pass'] and bool(rat) and max(rat) >= 1.03
    return out


def confirm(cell, a1, b1, b2, a2, bands):
    A1, B1, B2, A2 = (retained(x) for x in (a1, b1, b2, a2))
    assert len(A1) == len(B1) == len(B2) == len(A2) == 5
    res = paired(A1 + A2, B1 + B2)  # pairs (A1[i], B1[i]) and (A2[i], B2[i])
    lo, hi = json.loads(pathlib.Path(bands).read_text())['cells'][cell]['band_drafts_on']
    a_loads = [describe(A1), describe(A2)]
    res.update(cell=cell, A_loads=a_loads, B_loads=[describe(B1), describe(B2)],
               A_in_band=[lo <= x['mean'] <= hi for x in a_loads], A_cv_pass=[x['cv'] <= 0.015 for x in a_loads])
    res['admit'] = res['t95'][0] > 1.05 and all(res['A_in_band']) and all(res['A_cv_pass'])
    return res


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'summary':
        r = {p: {m: describe([x['metric'] for x in rounds(p, m) if not x['warmup'] and x['ids_pass']]) for m in ('drafted', 'plain') if rounds(p, m)} for p in args}
    elif cmd == 'phase0':
        r = phase0(args)
    elif cmd == 'screen':
        r = screen(json.loads(pathlib.Path(args[0]).read_text()), args[1], args[2:])
    elif cmd == 'confirm':
        r = confirm(*args)
    else:
        raise SystemExit(__doc__)
    print(json.dumps(r, indent=2))
