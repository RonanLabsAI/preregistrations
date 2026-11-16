#!/usr/bin/env python3
"""Study D analysis. All intervals two-sided Student t95 on paired log ratios, back-transformed.

analyze_d.py h CELLS_DIR            -> head-to-head (TensorFold champion vs SGLang daily copy)
analyze_d.py w CELLS_DIR            -> TensorFold workloads: lossless, long decode, temperature, cache, prefill screen
analyze_d.py wconfirm CELLS_DIR ARM -> prefill-arm confirm (W-B / ARM-1 / ARM-2 / W-Bn)
"""
import json, math, pathlib, sys

T = {2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582, 7: 2.364624, 9: 2.262157}


def rounds(p, retained=True):
    rs = [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]
    rs = [r for r in rs if 'metric' in r]
    return [r for r in rs if not r.get('warmup')] if retained else rs


def ok(p):
    return all(r.get('ok') for r in rounds(p, False))


def paired(a, b):
    """ratio b/a over pairs."""
    logs = [math.log(y / x) for x, y in zip(a, b)]
    n = len(logs)
    m = sum(logs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in logs) / (n - 1)) if n > 1 else float('nan')
    h = T[n - 1] * sd / math.sqrt(n)
    return dict(n=n, ratio=math.exp(m), t95=[math.exp(m - h), math.exp(m + h)])


def mean(v):
    return sum(v) / len(v)


def native_decode(p):
    out = []
    for r in rounds(p):
        v = [(x['completion_tokens'] - 1) / x['native_seconds'] for x in r['rows'] if x.get('native_seconds')]
        out.append(mean(v) if v else None)
    return out


def wilson(k, n):
    z = 1.959964
    c = (k + z * z / 2) / (n + z * z)
    h = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return [c - h, c + h]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def h(cells):
    d = pathlib.Path(cells)
    f = lambda tag, name: d / ('%s-%s.jsonl' % (tag, name))
    out = dict(speed={}, prefill={}, long256={}, quality=None, ids_pass_tf=True)
    for cell in ('C1', 'C4', 'C16'):
        nm = 'speed-' + cell
        if not all(f(t, nm).exists() for t in ('H-T1', 'H-S1', 'H-T2', 'H-S2')):
            continue
        out['ids_pass_tf'] &= ok(f('H-T1', nm)) and ok(f('H-T2', nm))
        S = [r['metric'] for t in ('H-S1', 'H-S2') for r in rounds(f(t, nm))]
        Tv = [r['metric'] for t in ('H-T1', 'H-T2') for r in rounds(f(t, nm))]
        out['speed'][cell] = dict(sglang_mean=mean(S), tf_mean=mean(Tv), tf_over_sglang=paired(S, Tv))
    for task in ('prefill', 'long256'):
        for s in ('L8K', 'L30K'):
            nm = '%s-C1%s' % (task, s)
            if not all(f(t, nm).exists() for t in ('H-T1', 'H-S1', 'H-T2', 'H-S2')):
                continue
            S = [r['metric'] for t in ('H-S1', 'H-S2') for r in rounds(f(t, nm))]
            Tv = [r['metric'] for t in ('H-T1', 'H-T2') for r in rounds(f(t, nm))]
            out[task][s] = dict(sglang_mean=mean(S), tf_mean=mean(Tv), tf_over_sglang=paired(S, Tv))
    qa, qb = f('H-T1', 'gsm8k-'), f('H-S1', 'gsm8k-')
    if qa.exists() and qb.exists():
        A = {x['id']: x['correct'] for x in json.loads(qa.read_text().splitlines()[0])['rows']}
        B = {x['id']: x['correct'] for x in json.loads(qb.read_text().splitlines()[0])['rows']}
        ids = sorted(A)
        n = len(ids)
        ka, kb = sum(A[i] for i in ids), sum(B[i] for i in ids)
        b = sum(A[i] and not B[i] for i in ids)
        c = sum(B[i] and not A[i] for i in ids)
        diff = (ka - kb) / n
        se = math.sqrt(max((b + c) - (b - c) ** 2 / n, 0)) / n
        out['quality'] = dict(n=n, tf_accuracy=ka / n, tf_wilson95=wilson(ka, n), sglang_accuracy=kb / n, sglang_wilson95=wilson(kb, n),
                              tf_only_correct=b, sglang_only_correct=c, diff_tf_minus_sglang=diff,
                              diff_95=[diff - 1.959964 * se, diff + 1.959964 * se], mcnemar_p=mcnemar(b, c),
                              parity=abs(diff - 1.959964 * se) <= 0.03 and abs(diff + 1.959964 * se) <= 0.03)
    return out


def w(cells):
    d = pathlib.Path(cells)
    f = lambda tag, name: d / ('%s-%s.jsonl' % (tag, name))
    out = dict(lossless={}, long_native_decode={}, temperature={}, cache=None, screen={})
    for tag in ('W-A', 'W-B'):
        for s in ('L8K', 'L30K'):
            p = f(tag, 'long256-C1' + s)
            if p.exists():
                out['lossless']['%s %s' % (tag, s)] = ok(p)
    for s in ('L8K', 'L30K'):
        a, b = f('W-A', 'long256-C1' + s), f('W-B', 'long256-C1' + s)
        if a.exists() and b.exists():
            A, B = native_decode(a), native_decode(b)
            out['long_native_decode'][s] = dict(A_mean=mean(A), D4b_mean=mean(B), D4b_over_A_descriptive=mean(B) / mean(A))
    for cell in ('C1', 'C16'):
        a, b = f('W-A', 'temp-' + cell), f('W-B', 'temp-' + cell)
        if a.exists() and b.exists():
            A, B = [r['metric'] for r in rounds(a)], [r['metric'] for r in rounds(b)]
            out['temperature'][cell] = dict(A_mean=mean(A), D4b_mean=mean(B), D4b_over_A_descriptive=mean(B) / mean(A))
    c = f('W-C', 'cache-C1')
    if c.exists():
        rs = rounds(c)
        out['cache'] = dict(cold_over_warm=[r['metric'] for r in rs], cold_seconds=[r['cold_seconds'] for r in rs],
                            warm_seconds=[r['warm_seconds'] for r in rs])
    base_p, base_m = f('W-B', 'prefill-C1L30K'), f('W-B', 'mixed-C1')
    if base_p.exists() and base_m.exists():
        bp = mean([r['metric'] for r in rounds(base_p)])
        bm = rounds(base_m)
        bl, bs = mean([r['long_tok_s'] for r in bm]), mean([r['short_tok_s'] for r in bm])
        for arm in ('S2', 'S3', 'S4', 'S5', 'S6'):
            p, m = f('W-' + arm, 'prefill-C1L30K'), f('W-' + arm, 'mixed-C1')
            if not (p.exists() and m.exists()):
                continue
            mm = rounds(m)
            r = dict(ids_pass=ok(m) and ok(p), prefill_ratio=mean([x['metric'] for x in rounds(p)]) / bp,
                     mixed_long_ratio=mean([x['long_tok_s'] for x in mm]) / bl, mixed_short_ratio=mean([x['short_tok_s'] for x in mm]) / bs)
            r['advance'] = r['ids_pass'] and max(r['prefill_ratio'], r['mixed_long_ratio']) >= 1.03 and r['mixed_short_ratio'] >= 0.97
            out['screen'][arm] = r
    return out


def wconfirm(cells, arm, a1, a2):
    d = pathlib.Path(cells)
    f = lambda tag, name: d / ('%s-%s.jsonl' % (tag, name))
    res = {}
    for name, key in (('prefill-C1L30K', 'metric'), ('mixed-C1', 'long_tok_s'), ('mixed-C1', 'short_tok_s')):
        A = [r[key] for t in (a1, a2) for r in rounds(f(t, name))]
        B = [r[key] for t in ('WC-%s-1' % arm, 'WC-%s-2' % arm) for r in rounds(f(t, name))]
        res['%s:%s' % (name, key)] = paired(A, B)
    ids = all(ok(f(t, n)) for t in ('WC-%s-1' % arm, 'WC-%s-2' % arm) for n in ('prefill-C1L30K', 'mixed-C1'))
    gain = max(res['prefill-C1L30K:metric']['t95'][0], res['mixed-C1:long_tok_s']['t95'][0])
    res.update(arm=arm, ids_pass=ids, admit=ids and gain > 1.05 and res['mixed-C1:short_tok_s']['t95'][0] >= 0.97)
    return res


if __name__ == '__main__':
    cmd = sys.argv[1]
    r = h(sys.argv[2]) if cmd == 'h' else w(sys.argv[2]) if cmd == 'w' else wconfirm(*sys.argv[2:6])
    print(json.dumps(r, indent=2))
