import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

from laya_shim.server import Handler


class Shim:
    """A laya-shim server on a free local port, answering with `agent`."""

    def __init__(self, agent):
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.server.agent = agent
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, body, path="/v1/systemone"):
        """Sends `body` the way omp's TypeSafe client does; returns (status, JSON)."""
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        request = urllib.request.Request(
            self.url + path,
            data=data,
            headers={
                "Authorization": "Bearer laya-local",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
