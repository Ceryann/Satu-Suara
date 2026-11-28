"""WebSocket penerima: hello/welcome, ping/pong, Tes Bunyi, audio (M3)."""
import asyncio
import itertools
import json
import math
import time
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .audio import CHANNELS, CHUNK_MS, RATE
from .sync import now_ms

FORMAT = "s16le"
DEFAULT_DELAY_MS = 500
HELLO_TIMEOUT_S = 10
KICK_CODE = 4403            # kode tutup khusus: klien tahu ini diputus host dan tidak menyambung otomatis
MAX_PENDING_AUDIO = 50      # ~2 detik; klien yang tertinggal lebih jauh dilewati chunk-nya


def detect_kind(ua: str) -> str:
    u = (ua or "").lower()
    return "HP" if ("android" in u or "mobile" in u) else "Laptop"


_STATUS_NUM = {"buf_ms": (0, 600000), "min_buf_ms": (-60000, 600000), "late": (0, 10**9), "stalls": (0, 10**9),
               "played": (0, 10**9), "rtt_ms": (0, 600000), "reconnects": (0, 10**6),
               "trim_ppm": (-100000, 100000), "jumps": (0, 10**9)}
_STATUS_ENUM = {"vis": {"visible", "hidden"}, "wake": {"api", "video", "none"},
                "ctx": {"running", "suspended", "interrupted", "closed"}}


def clean_status(msg: dict) -> dict:
    """Laporan dari klien tidak dipercaya: hanya kunci dikenal, angka dibatasi, teks dari daftar tetap."""
    out: dict = {}
    for key, (lo, hi) in _STATUS_NUM.items():
        v = msg.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[key] = int(max(lo, min(hi, round(v))))
        elif v is None and key in msg and key in ("min_buf_ms", "rtt_ms"):
            out[key] = None
    for key, allowed in _STATUS_ENUM.items():
        if isinstance(msg.get(key), str) and msg[key] in allowed:
            out[key] = msg[key]
    for key in ("synced", "streaming"):
        if isinstance(msg.get(key), bool):
            out[key] = msg[key]
    return out


@dataclass
class Client:
    id: int
    name: str
    kind: str
    ip: str
    ws: WebSocket = field(repr=False, default=None)
    connected_at: float = field(default_factory=time.time)
    out: asyncio.PriorityQueue = field(default_factory=asyncio.PriorityQueue, repr=False)
    pending_audio: int = 0
    dropped_audio: int = 0
    volume: int = 100                 # diatur dari dashboard host, berlaku di atas volume lokal klien
    muted: bool = False
    key: str = ""
    kicked: bool = False              # diputus oleh host (bukan putus sendiri)
    status: dict = field(default_factory=dict)   # laporan berkala dari klien (buffer, jeda, layar, ...)
    status_at: float = 0.0
    _n: itertools.count = field(default_factory=itertools.count, repr=False)

    def send_json(self, msg: dict) -> None:
        """Pesan kontrol (prioritas 0): mendahului audio agar pong tidak tertunda."""
        self.out.put_nowait((0, next(self._n), "t", json.dumps(msg)))

    def close(self, code: int = KICK_CODE, reason: str = "diputus_host") -> None:
        """Minta penulis tunggal menutup socket (prioritas 0: mendahului audio yang masih antre)."""
        self.out.put_nowait((0, next(self._n), "c", (code, reason)))

    def send_audio(self, data: bytes) -> bool:
        if self.pending_audio >= MAX_PENDING_AUDIO:
            self.dropped_audio += 1
            return False
        self.pending_audio += 1
        self.out.put_nowait((1, next(self._n), "b", data))
        return True


class ClientManager:
    def __init__(self) -> None:
        self._clients: dict[int, Client] = {}
        self._next_id = 1
        self._kind_count = {"HP": 0, "Laptop": 0}
        self._names: dict[str, str] = {}   # device_id -> nama, agar nama tetap saat reconnect
        self._controls: dict[str, dict] = {}   # device_id -> {volume, muted}, tetap saat reconnect
        self.delay_ms = DEFAULT_DELAY_MS       # delay target saat ini (diubah oleh AudioEngine.set_delay)

    def add(self, ws, ua: str, ip: str, device_id: str | None, name: str | None) -> Client:
        kind = detect_kind(ua)
        key = (str(device_id)[:64] if device_id else f"anon-{self._next_id}")
        label = self._names.get(key)
        if not label:
            self._kind_count[kind] += 1
            label = (name or "").strip()[:40] or f"{kind} {self._kind_count[kind]}"
            self._names[key] = label
        ctl = self._controls.get(key, {})
        client = Client(id=self._next_id, name=label, kind=kind, ip=ip, ws=ws,
                        volume=ctl.get("volume", 100), muted=ctl.get("muted", False), key=key)
        self._next_id += 1
        self._clients[client.id] = client
        return client

    def remove(self, client_id: int) -> None:
        self._clients.pop(client_id, None)

    def count(self) -> int:
        return len(self._clients)

    def set_control(self, client_id: int, volume=None, muted=None) -> bool:
        c = self._clients.get(client_id)
        if c is None:
            return False
        if volume is not None:
            c.volume = int(max(0, min(100, volume)))
        if muted is not None:
            c.muted = bool(muted)
        self._controls[c.key] = {"volume": c.volume, "muted": c.muted}
        c.send_json({"type": "control", "volume": c.volume, "muted": c.muted})
        return True

    async def broadcast_json(self, msg: dict) -> None:
        for c in list(self._clients.values()):
            c.send_json(msg)

    def broadcast_audio(self, data: bytes) -> None:
        for c in list(self._clients.values()):
            c.send_audio(data)

    def kick(self, client_id: int) -> bool:
        """Putuskan satu klien. Langsung dikeluarkan dari daftar sehingga audio berhenti dikirim."""
        c = self._clients.pop(client_id, None)
        if c is None:
            return False
        c.kicked = True
        c.close()
        return True

    def kick_all(self) -> int:
        ids = list(self._clients)
        return sum(1 for i in ids if self.kick(i))

    def snapshot(self) -> list[dict]:
        now = time.time()
        return [
            {"id": c.id, "name": c.name, "kind": c.kind, "ip": c.ip,
             "connected_s": int(now - c.connected_at), "dropped_audio": c.dropped_audio,
             "volume": c.volume, "muted": c.muted,
             "status": c.status or None, "status_age_s": round(now - c.status_at, 1) if c.status_at else None}
            for c in self._clients.values()
        ]


async def run_sender(client: Client) -> None:
    """Satu-satunya penulis ke socket klien: pesan kontrol lebih dulu, lalu audio."""
    try:
        while True:
            _, _, kind, payload = await client.out.get()
            if kind == "t":
                await client.ws.send_text(payload)
            elif kind == "c":
                await client.ws.close(code=payload[0], reason=payload[1])
                return
            else:
                client.pending_audio -= 1
                await client.ws.send_bytes(payload)
    except Exception:
        return  # koneksi putus: handler utama yang membersihkan


def make_ws_router(session, manager: ClientManager, click_test) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket, t: str = ""):
        await websocket.accept()
        if not session.valid_token(t):
            # Diterima dulu lalu ditutup 4401 agar klien bisa membedakan
            # "token kedaluwarsa" dari "jaringan putus".
            await websocket.close(code=4401, reason="token tidak valid")
            return
        client = None
        sender = None
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), HELLO_TIMEOUT_S)
            hello = json.loads(raw)
            if not isinstance(hello, dict) or hello.get("type") != "hello":
                await websocket.close(code=4400, reason="hello diharapkan")
                return
            ip = websocket.client.host if websocket.client else "?"
            client = manager.add(websocket, hello.get("ua", ""), ip, hello.get("device_id"), hello.get("name"))
            await websocket.send_json({
                "type": "welcome",
                "client_id": client.id,
                "name": client.name,
                "kind": client.kind,
                "rate": RATE,
                "channels": CHANNELS,
                "format": FORMAT,
                "chunk_ms": CHUNK_MS,
                "delay_ms": manager.delay_ms,
                "volume": client.volume,
                "muted": client.muted,
            })
            sender = asyncio.create_task(run_sender(client))
            print(f"  + {client.name} tersambung ({ip}). Total: {manager.count()}", flush=True)
            while True:
                raw = await websocket.receive_text()
                ts = now_ms()  # cap waktu secepat mungkin setelah pesan tiba (untuk pong)
                msg = json.loads(raw)
                kind = msg.get("type") if isinstance(msg, dict) else None
                if kind == "ping":
                    client.send_json({"type": "pong", "t0": msg.get("t0"), "ts": ts})
                elif kind == "test":
                    if click_test.start():
                        print(f"  * Tes Bunyi dimulai oleh {client.name}", flush=True)
                elif kind == "status":
                    client.status = clean_status(msg)
                    client.status_at = time.time()
        except (WebSocketDisconnect, asyncio.TimeoutError, ValueError, KeyError, RuntimeError):
            pass
        finally:
            if sender is not None:
                sender.cancel()
            if client is not None:
                manager.remove(client.id)
                extra = f", {client.dropped_audio} chunk dilewati" if client.dropped_audio else ""
                how = "diputus oleh host" if client.kicked else "putus"
                print(f"  - {client.name} {how}. Total: {manager.count()}{extra}", flush=True)

    return router
