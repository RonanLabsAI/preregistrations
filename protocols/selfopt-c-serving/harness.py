#!/usr/bin/env python3
"""Study C client: one cell on one load, alternating request modes, bitwise token-ID gate.

Usage: harness.py --url http://127.0.0.1:PORT --model NAME --corpus reference-corpus.json
       --reference TOKEN-REFERENCE.json --out cell.jsonl --cell C1|C4|C16
       --modes drafted[,plain] --rounds N [--stop-file PATH]
Writes one JSON line per round (warm rounds have repeat -1) and exits 3 at the first ID mismatch.
"""
import argparse, hashlib, json, pathlib, sys, threading, time, urllib.error, urllib.request
from concurrent import futures
from datetime import datetime
from zoneinfo import ZoneInfo

CLIENTS = {'C1': 1, 'C4': 4, 'C16': 16}


def stamp():
    return datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()


def ids_sha(ids):
    return hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()


class Mismatch(Exception):
    pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cell', required=True, choices=sorted(CLIENTS))
    ap.add_argument('--modes', default='drafted')
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--stop-file')
    ap.add_argument('--timeout', type=float, default=900)
    a = ap.parse_args(argv)
    prompts = json.loads(pathlib.Path(a.corpus).read_text())['prompts']
    assert len(prompts) == 16
    ref = json.loads(pathlib.Path(a.reference).read_text())['long_plain_decode']
    modes = a.modes.split(',')
    assert set(modes) <= {'drafted', 'plain'} and modes
    out = pathlib.Path(a.out)
    raw = out.with_suffix('.raw.jsonl')
    lock = threading.Lock()

    def post(i, mode):
        p = prompts[i]
        body = dict(model=a.model, messages=[dict(role='user', content=p['prompt'])], temperature=0,
                    seed=1234, max_tokens=512, draft=(mode == 'drafted'), return_token_ids=True,
                    chat_template_kwargs=dict(enable_thinking=False))
        req = urllib.request.Request(a.url.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            payload = dict(http_status=e.code, body=e.read().decode(errors='replace'))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            payload = dict(error=type(e).__name__, reason=str(e))
        client = time.monotonic() - t0
        with lock, raw.open('a') as f:
            f.write(json.dumps(dict(pt=stamp(), id=p['id'], mode=mode, client_seconds=client, response=payload)) + '\n')
        tf = payload.get('tensorfold') or {}
        u = payload.get('usage') or {}
        ids = tf.get('token_ids')
        problems = []
        if not isinstance(ids, list):
            problems.append('no token_ids')
            ids = []
        if u.get('completion_tokens') != 512 or len(ids) != 512:
            problems.append('completion %s/%s' % (u.get('completion_tokens'), len(ids)))
        if (u.get('prompt_tokens_details') or {}).get('cached_tokens') != 0:
            problems.append('cached tokens')
        if bool(tf.get('drafts')) != (mode == 'drafted'):
            problems.append('drafts flag %r' % tf.get('drafts'))
        if tf.get('enable_thinking'):
            problems.append('thinking on')
        if ids and tf.get('token_sha256') != ids_sha(ids):
            problems.append('token_sha256 differs from comma-joined IDs')
        first_diff = None
        if ids != ref[p['id']]:
            first_diff = next((k for k, (x, y) in enumerate(zip(ids, ref[p['id']])) if x != y), min(len(ids), 512))
            problems.append('ID mismatch at %d' % first_diff)
        spec = payload.get('speculative') or tf.get('speculative') or {k: tf[k] for k in ('drafted', 'accepted', 'rounds') if k in tf}
        return dict(id=p['id'], mode=mode, native_seconds=tf.get('seconds'), client_seconds=client,
                    ttft=tf.get('time_to_first_token'), token_sha256=ids_sha(ids) if ids else None,
                    prompt_tokens=u.get('prompt_tokens'), speculative=spec, problems=problems,
                    first_diff=first_diff)

    def one_round(mode, repeat):
        n = CLIENTS[a.cell]
        started = stamp()
        t0 = time.monotonic()
        if n == 1:
            rows = [post(i, mode) for i in range(16)]
        else:
            per = 16 // n

            def client(k):
                return [post(i, mode) for i in range(k * per, (k + 1) * per)]
            with futures.ThreadPoolExecutor(max_workers=n) as pool:
                rows = [r for part in pool.map(client, range(n)) for r in part]
        wall = time.monotonic() - t0
        ok = all(not r['problems'] for r in rows)
        native = sum(511 / r['native_seconds'] for r in rows) / 16 if ok else None
        return dict(pt=started, cell=a.cell, mode=mode, repeat=repeat, warmup=repeat < 0, wall_seconds=wall,
                    native_reply_tok_s=native, aggregate_client_tok_s=8192 / wall if ok else None,
                    metric=(native if a.cell == 'C1' else (8192 / wall if ok else None)), ids_pass=ok,
                    requests=rows)

    with out.open('x') as f:
        for repeat in range(-1, a.rounds):
            for mode in modes:
                if a.stop_file and pathlib.Path(a.stop_file).exists():
                    f.write(json.dumps(dict(pt=stamp(), stopped='stop file', cell=a.cell, mode=mode, repeat=repeat)) + '\n')
                    return 4
                r = one_round(mode, repeat)
                f.write(json.dumps(r) + '\n')
                f.flush()
                print(json.dumps({k: r[k] for k in ('cell', 'mode', 'repeat', 'metric', 'wall_seconds', 'ids_pass')}), flush=True)
                if not r['ids_pass']:
                    bad = [dict(id=x['id'], problems=x['problems']) for x in r['requests'] if x['problems']]
                    print(json.dumps(dict(gate='FAIL', cell=a.cell, mode=mode, repeat=repeat, bad=bad)), flush=True)
                    return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
