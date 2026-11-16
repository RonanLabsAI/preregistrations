#!/usr/bin/env python3
"""Study D client, engine-neutral (TensorFold or SGLang), one task on one load.

Tasks:
  speed     16 short reference prompts, 512 tokens, cell C1|C4|C16. Metric: completion tokens / round wall s.
            tf: IDs must equal --reference (exit 3 on mismatch). sglang: ignore_eos, 512 tokens required.
  prefill   long prompts (--set L8K|L30K), max_tokens 1, one at a time. Metric: mean prompt tok / client s.
  long256   long prompts, max_tokens 256, one at a time. Metric: mean completion tok / client s (incl. prefill).
            With --reference (tf), IDs must match; with --write-reference, IDs are saved.
  mixed     C4: client 0 sends the 4 L30K prompts (max_tokens 64); clients 1-3 send short prompts 4..15 (512).
            Metrics: long_tok_s = mean (prompt+completion) / client s of long requests; short_tok_s = 6144 / short wall.
  gsm8k     500 GSM8K test problems, 16 concurrent, max_tokens 1024. One pass. Records answers and correctness.
  temp      short prompts at temperature 0.7, top_p 0.95, cell C1|C16. Metric: completion tokens / wall.
  cache     4 L30K variants sharing one context (final question differs), one at a time, max_tokens 1.
--nonce prefixes each long prompt with a fresh session line so no engine can reuse a cached prefix.
For sglang, POST /flush_cache runs before every round.
"""
import argparse, hashlib, json, pathlib, re, secrets, sys, threading, time, urllib.error, urllib.request
from concurrent import futures
from datetime import datetime
from zoneinfo import ZoneInfo

CLIENTS = {'C1': 1, 'C4': 4, 'C16': 16}
NUM = re.compile(r'-?\d[\d,]*(?:\.\d+)?')


def stamp():
    return datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()


def sha(ids):
    return hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()


def extract(text):
    m = re.findall(r'####\s*\$?\s*(-?[\d,]*\.?\d+)', text or '')
    v = m[-1] if m else (NUM.findall(text or '') or [None])[-1]
    if v is None:
        return None
    v = v.replace(',', '').rstrip('.')
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True, choices=['tf', 'sglang'])
    ap.add_argument('--url', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--task', required=True, choices=['speed', 'prefill', 'long256', 'mixed', 'gsm8k', 'temp', 'cache'])
    ap.add_argument('--short-corpus', required=True)
    ap.add_argument('--corpus-d', required=True)
    ap.add_argument('--reference')
    ap.add_argument('--write-reference')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cell', default='C1')
    ap.add_argument('--set', default='L30K')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--draft', default='true')
    ap.add_argument('--nonce', action='store_true')
    ap.add_argument('--stop-file')
    ap.add_argument('--timeout', type=float, default=1800)
    a = ap.parse_args(argv)
    short = json.loads(pathlib.Path(a.short_corpus).read_text())['prompts']
    cd = json.loads(pathlib.Path(a.corpus_d).read_text())
    longp = [p for p in cd['long_prompts'] if p['id'].startswith(a.set + '-')]
    ref = json.loads(pathlib.Path(a.reference).read_text()) if a.reference else None
    draft = a.draft == 'true'
    out = pathlib.Path(a.out)
    raw = out.with_suffix('.raw.jsonl')
    lock = threading.Lock()
    newref = {}

    def post(pid, content, max_tokens, temperature=0.0, top_p=None, check=None, force_len=None):
        body = dict(model=a.model, messages=[dict(role='user', content=content)], temperature=temperature,
                    seed=1234, max_tokens=max_tokens, chat_template_kwargs=dict(enable_thinking=False))
        if top_p is not None:
            body['top_p'] = top_p
        if a.engine == 'tf':
            body.update(draft=draft, return_token_ids=True)
        elif force_len:
            body['ignore_eos'] = True
        req = urllib.request.Request(a.url.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            payload = dict(http_status=e.code, body=e.read().decode(errors='replace')[:2000])
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            payload = dict(error=type(e).__name__, reason=str(e))
        client = time.monotonic() - t0
        with lock, raw.open('a') as f:
            f.write(json.dumps(dict(pt=stamp(), id=pid, client_seconds=client, response=payload)) + '\n')
        u = payload.get('usage') or {}
        tf = payload.get('tensorfold') or {}
        msg = ((payload.get('choices') or [{}])[0].get('message') or {})
        ids = tf.get('token_ids') if a.engine == 'tf' else None
        problems = []
        if 'choices' not in payload:
            problems.append('no reply: %s' % (payload.get('http_status') or payload.get('error')))
        if force_len and u.get('completion_tokens') != force_len:
            problems.append('completion %s != %s' % (u.get('completion_tokens'), force_len))
        cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
        if a.engine == 'tf' and a.task != 'cache' and cached:
            problems.append('cached tokens %s' % cached)
        if a.engine == 'tf' and tf.get('enable_thinking'):
            problems.append('thinking on')
        first_diff = None
        if check is not None and ids is not None and ids != check:
            first_diff = next((k for k, (x, y) in enumerate(zip(ids, check)) if x != y), min(len(ids), len(check)))
            problems.append('ID mismatch at %d' % first_diff)
        return dict(id=pid, client_seconds=client, prompt_tokens=u.get('prompt_tokens'), completion_tokens=u.get('completion_tokens'),
                    cached_tokens=cached, native_seconds=tf.get('seconds'), ttft_native=tf.get('time_to_first_token'),
                    token_sha256=sha(ids) if ids else None, ids=ids, content=msg.get('content'), problems=problems,
                    first_diff=first_diff, speculative=payload.get('speculative') or tf.get('speculative'))

    def flush():
        if a.engine == 'sglang':
            try:
                urllib.request.urlopen(urllib.request.Request(a.url.rstrip('/') + '/flush_cache', data=b'', method='POST'), timeout=60).read()
            except Exception as e:
                print(json.dumps(dict(warn='flush_cache failed', err=str(e))), flush=True)

    def lp(p):
        return ('[session %s]\n' % secrets.token_hex(4) if a.nonce else '') + p['prompt']

    def run_round(repeat):
        flush()
        started, t0 = stamp(), time.monotonic()
        if a.task in ('speed', 'temp'):
            n = CLIENTS[a.cell]
            per = 16 // n
            temp = a.task == 'temp'

            def client(k):
                rows = []
                for i in range(k * per, (k + 1) * per):
                    p = short[i]
                    chk = ref['long_plain_decode'][p['id']] if (ref and not temp and a.engine == 'tf') else None
                    rows.append(post(p['id'], p['prompt'], 512, 0.7 if temp else 0.0, 0.95 if temp else None, chk,
                                     None if temp else 512))
                return rows
            with futures.ThreadPoolExecutor(max_workers=n) as pool:
                rows = [r for part in pool.map(client, range(n)) for r in part]
            wall = time.monotonic() - t0
            toks = sum(r['completion_tokens'] or 0 for r in rows)
            return dict(metric=toks / wall, wall_seconds=wall, completion_tokens=toks, rows=rows)
        if a.task in ('prefill', 'long256'):
            mt = 1 if a.task == 'prefill' else 256
            rows = []
            for p in longp:
                chk = ref.get(p['id']) if ref else None
                rows.append(post(p['id'], lp(p), mt, check=chk))
            vals = [(r['prompt_tokens'] or 0) / r['client_seconds'] if a.task == 'prefill' else (r['completion_tokens'] or 0) / r['client_seconds'] for r in rows]
            if a.write_reference and a.task == 'long256':
                for r in rows:
                    if r['id'] in newref and newref[r['id']] != r['ids']:
                        r['problems'].append('reference rounds disagree')
                    newref[r['id']] = r['ids']
            return dict(metric=sum(vals) / len(vals), per_prompt=vals, wall_seconds=time.monotonic() - t0, rows=rows)
        if a.task == 'mixed':
            l30 = [p for p in cd['long_prompts'] if p['id'].startswith('L30K-')]
            res = {}

            def longc():
                return [post(p['id'], lp(p), 64) for p in l30]

            def shortc(k):
                t = time.monotonic()
                rows = []
                for i in range(4 + 4 * k, 8 + 4 * k):
                    p = short[i]
                    chk = ref['long_plain_decode'][p['id']] if (ref and a.engine == 'tf') else None
                    rows.append(post(p['id'], p['prompt'], 512, check=chk, force_len=512))
                return rows, time.monotonic() - t
            with futures.ThreadPoolExecutor(max_workers=4) as pool:
                fl = pool.submit(longc)
                fs = [pool.submit(shortc, k) for k in range(3)]
                lrows = fl.result()
                srs = [f.result() for f in fs]
            srows = [r for rr, _ in srs for r in rr]
            swall = max(w for _, w in srs)
            ltoks = [((r['prompt_tokens'] or 0) + (r['completion_tokens'] or 0)) / r['client_seconds'] for r in lrows]
            return dict(metric=sum(ltoks) / len(ltoks), long_tok_s=sum(ltoks) / len(ltoks),
                        short_tok_s=sum(r['completion_tokens'] or 0 for r in srows) / swall,
                        wall_seconds=time.monotonic() - t0, rows=lrows + srows)
        if a.task == 'cache':
            base = [p for p in cd['long_prompts'] if p['id'] == 'L30K-0'][0]['prompt']
            ctx = base[:base.index('\nNow solve this new problem.')]
            finals = [p['prompt'][p['prompt'].index('\nNow solve this new problem.'):] for p in cd['long_prompts'] if p['id'].startswith('L30K-')]
            tagline = '[session %s]\n' % secrets.token_hex(4)
            rows = [post('cache-%d' % k, tagline + ctx + f, 1) for k, f in enumerate(finals)]
            return dict(metric=rows[0]['client_seconds'] / (sum(r['client_seconds'] for r in rows[1:]) / 3),
                        cold_seconds=rows[0]['client_seconds'], warm_seconds=[r['client_seconds'] for r in rows[1:]],
                        wall_seconds=time.monotonic() - t0, rows=rows)
        raise SystemExit('bad task')

    def gsm8k():
        q = cd['quality']
        inst = cd['quality_instruction']

        def one(item):
            r = post(item['id'], inst + '\nProblem: ' + item['question'], 1024)
            got = extract(r['content'])
            r.update(gold=item['answer'], extracted=got, correct=(got is not None and got == extract('#### ' + item['answer'])))
            r.pop('ids', None)
            return r
        flush()
        t0 = time.monotonic()
        with futures.ThreadPoolExecutor(max_workers=16) as pool:
            rows = list(pool.map(one, q))
        acc = sum(r['correct'] for r in rows) / len(rows)
        return dict(metric=acc, accuracy=acc, n=len(rows), wall_seconds=time.monotonic() - t0, rows=rows)

    with out.open('x') as f:
        if a.task == 'gsm8k':
            r = gsm8k()
            r.update(pt=stamp(), task='gsm8k', engine=a.engine, ok=all(not x['problems'] for x in r['rows']))
            f.write(json.dumps(r) + '\n')
            print(json.dumps(dict(task='gsm8k', engine=a.engine, accuracy=r['accuracy'], ok=r['ok'])), flush=True)
            return 0 if r['ok'] else 5
        for repeat in range(-1, a.rounds):
            if a.stop_file and pathlib.Path(a.stop_file).exists():
                f.write(json.dumps(dict(pt=stamp(), stopped='stop file', repeat=repeat)) + '\n')
                return 4
            r = run_round(repeat)
            ok = all(not x['problems'] for x in r['rows'])
            for x in r['rows']:
                x.pop('ids', None)
            r.update(pt=stamp(), task=a.task, engine=a.engine, cell=a.cell, set=a.set, repeat=repeat, warmup=repeat < 0, ok=ok)
            f.write(json.dumps(r) + '\n')
            f.flush()
            print(json.dumps({k: r.get(k) for k in ('task', 'engine', 'cell', 'set', 'repeat', 'metric', 'short_tok_s', 'wall_seconds', 'ok')}), flush=True)
            if not ok:
                bad = [dict(id=x['id'], problems=x['problems']) for x in r['rows'] if x['problems']]
                print(json.dumps(dict(gate='FAIL', bad=bad[:8])), flush=True)
                return 3
    if a.write_reference:
        pathlib.Path(a.write_reference).write_text(json.dumps(newref) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
