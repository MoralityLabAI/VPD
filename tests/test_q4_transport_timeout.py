"""Exercise a delayed completion response without invoking a language model."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from experiments.nanbeige_reasoning.run_q4_reasoning import request


def test_longer_transport_wait_preserves_completion_payload():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            time.sleep(.3)
            body = json.dumps({'tokens': [6, 7], 'stop_type': 'eos'}).encode()
            try:
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    payload = dict(prompt=[1, 2], n_predict=8192, temperature=0, seed=42,
                   repeat_penalty=1.0, return_tokens=True, stream=False,
                   cache_prompt=False, ignore_eos=False)
    try:
        base = f'http://127.0.0.1:{server.server_port}'
        with pytest.raises(TimeoutError):
            request(base, 'test-only', '/completion', payload, timeout=.05)
        result = request(base, 'test-only', '/completion', payload, timeout=3)
        assert result == {'tokens': [6, 7], 'stop_type': 'eos'}
        assert received == [payload, payload]
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
