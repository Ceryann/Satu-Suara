"""Token dan kode gabung per sesi (S-02). Berganti setiap server dijalankan."""
import secrets
import time
from collections import defaultdict, deque

MAX_FAILS = 5        # salah kode berturut-turut per IP...
WINDOW_S = 60        # ...dalam jendela waktu ini sebelum dikunci sementara


class Session:
    def __init__(self) -> None:
        self.token = secrets.token_urlsafe(9)
        self.code = f"{secrets.randbelow(10000):04d}"
        self._fails: dict[str, deque] = defaultdict(deque)

    def valid_token(self, token: str | None) -> bool:
        if not token:
            return False
        return secrets.compare_digest(token.encode(), self.token.encode())

    def try_code(self, ip: str, code: str):
        """Hasil: ("ok", token) | ("bad", None) | ("locked", detik_tunggu)."""
        now = time.monotonic()
        fails = self._fails[ip]
        while fails and now - fails[0] > WINDOW_S:
            fails.popleft()
        if len(fails) >= MAX_FAILS:
            return "locked", int(WINDOW_S - (now - fails[0])) + 1
        if secrets.compare_digest(code.encode(), self.code.encode()):
            fails.clear()
            return "ok", self.token
        fails.append(now)
        return "bad", None
