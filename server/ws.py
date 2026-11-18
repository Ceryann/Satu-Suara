"""WebSocket penerima (M1: hello/welcome, ping/pong, Tes Bunyi)."""
import asyncio
import json
import time
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .sync import now_ms

RATE = 48000
CHANNELS = 2
FORMAT = "s16le"
CHUNK_MS = 40
DEFAULT_DELAY_MS = 500
HELLO_TIMEOUT_S = 10


def detect_kind(ua: str) -> str:
    u = (ua or "").lower()
    return "HP" if ("android" in u or "mobile" in u) else "Laptop"


@dataclass
class Client:
    id: int
    name: str
    kind: str
    ip: str
    ws: WebSocket = field(repr=False, default=None)
    connected_at: float = field(default_factory=time.time)


class ClientManager:
    def __init__(self) -> None:
        self._clients: dict[int, Client] = {}
        self._next_id = 1
        self._kind_count = {"HP": 0, "Laptop": 0}
        self._names: dict[str, str] = {}   # device_id -> nama, agar nama tetap saat reconnect

    def add(self, ws, ua: str, ip: str, device_id: str | None, name: str | None) -> Client:
        kind = detect_kind(ua)
        key = (str(device_id)[:64] if device_id else f"anon-{self._next_id}")
        label = self._names.get(key)
        if not label:
            self._kind_count[kind] += 1
            label = (name or "").strip()[:40] or f"{kind} {self._kind_count[kind]}"
            self._names[key] = label
        client = Client(id=self._next_id, name=label, kind=kind, ip=ip, ws=ws)
        self._next_id += 1
        self._clients[client.id] = client
        return client

    def remove(self, client_id: int) -> None:
        self._clients.pop(client_id, None)

    def count(self) -> int:
        return len(self._clients)

    async def broadcast_json(self, msg: dict) -> None:
        clients = list(self._clients.values())
        results = await asyncio.gather(*(c.ws.send_json(msg) for c in clients), return_exceptions=True)
        _ = results  # klien yang putus akan dibersihkan oleh handler ws-nya sendiri

    def snapshot(self) -> list[dict]:
        now = time.time()
        return [
            {"id": c.id, "name": c.name, "kind": c.kind, "ip": c.ip,
             "connected_s": int(now - c.connected_at)}
            for c in self._clients.values()
        ]


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
                "delay_ms": DEFAULT_DELAY_MS,
            })
            print(f"  + {client.name} tersambung ({ip}). Total: {manager.count()}", flush=True)
            while True:
                raw = await websocket.receive_text()
                ts = now_ms()  # cap waktu secepat mungkin setelah pesan tiba (untuk pong)
                msg = json.loads(raw)
                kind = msg.get("type") if isinstance(msg, dict) else None
                if kind == "ping":
                    await websocket.send_json({"type": "pong", "t0": msg.get("t0"), "ts": ts})
                elif kind == "test":
                    if click_test.start():
                        print(f"  * Tes Bunyi dimulai oleh {client.name}", flush=True)
                # M6: status
        except (WebSocketDisconnect, asyncio.TimeoutError, ValueError, KeyError, RuntimeError):
            pass
        finally:
            if client is not None:
                manager.remove(client.id)
                print(f"  - {client.name} putus. Total: {manager.count()}", flush=True)

    return router
