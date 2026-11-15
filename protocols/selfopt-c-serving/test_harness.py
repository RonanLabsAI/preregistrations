#!/usr/bin/env python3
"""CPU-only self-test: a fake engine returns reference IDs (and one corrupted reply on request)."""
import hashlib, json, pathlib, subprocess, sys, threading, http.server, tempfile, time, random
here = pathlib.Path(__file__).resolve().parent
ref = json.loads((here / 'TOKEN-REFERENCE.json').read_text())['long_plain_decode']
corpus = {p['prompt']: p['id'] for p in json.loads((here / 'reference-corpus.json').read_text())['prompts']}
BAD = {'on': False}
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        ids = list(ref[corpus[b['messages'][0]['content']]])
        if BAD['on'] and b['draft']: ids[100] += 1
        s = 0.5 if b['draft'] else 1.0
        time.sleep(0.2 * s)
        body = dict(usage=dict(completion_tokens=512, prompt_tokens=90, prompt_tokens_details=dict(cached_tokens=0)),
                    tensorfold=dict(token_ids=ids, seconds=s * random.uniform(0.99, 1.01), drafts=b['draft'], enable_thinking=False,
                                    token_sha256=hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()))
        raw = json.dumps(body).encode(); self.send_response(200); self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
url = 'http://127.0.0.1:%d' % srv.server_port
with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d); paths = []
    for cell in ['C1', 'C4', 'C16']:
        o = d / (cell + '.jsonl'); paths.append(str(o))
        r = subprocess.run([sys.executable, str(here / 'harness.py'), '--url', url, '--model', 'x', '--corpus', str(here / 'reference-corpus.json'), '--reference', str(here / 'TOKEN-REFERENCE.json'), '--out', str(o), '--cell', cell, '--modes', 'drafted,plain', '--rounds', '5'], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    p0 = json.loads(subprocess.check_output([sys.executable, str(here / 'analyze.py'), 'phase0', *paths]))
    assert p0['baseline_pass'], p0
    a_over_r = p0['cells']['C1']['A_over_R']['ratio']; assert 1.9 < a_over_r < 2.1, a_over_r
    (d / 'p0.json').write_text(json.dumps(p0))
    s = json.loads(subprocess.check_output([sys.executable, str(here / 'analyze.py'), 'screen', str(d / 'p0.json'), 'X', *paths]))
    assert s['ids_pass'] and 0.97 < s['cells']['C1']['ratio'] < 1.03
    c = json.loads(subprocess.check_output([sys.executable, str(here / 'analyze.py'), 'confirm', 'C1', paths[0], paths[0], paths[0], paths[0], str(d / 'p0.json')]))
    assert c['n'] == 10 and abs(c['ratio'] - 1) < 1e-9 and not c['admit'] and all(c['A_in_band'])
    BAD['on'] = True
    o = d / 'bad.jsonl'
    r = subprocess.run([sys.executable, str(here / 'harness.py'), '--url', url, '--model', 'x', '--corpus', str(here / 'reference-corpus.json'), '--reference', str(here / 'TOKEN-REFERENCE.json'), '--out', str(o), '--cell', 'C4', '--modes', 'drafted', '--rounds', '3'], capture_output=True, text=True)
    assert r.returncode == 3 and 'ID mismatch at 100' in r.stdout, (r.returncode, r.stdout)
print('ALL STUDY C HARNESS SELF-TESTS PASS')
