"""SatuSuara: server, token, kode gabung, QR, jam server, Tes Bunyi, siaran audio PC."""
import argparse
import asyncio
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import qrcode
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .audio import AudioEngine, LoopbackSource, ToneSource
from .session import Session
from .sync import ClickTest
from .ws import DEFAULT_DELAY_MS, ClientManager, make_ws_router

DEFAULT_PORT = 8765
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
WEB_DIR = BASE_DIR / "web"
NO_STORE = {"Cache-Control": "no-store"}
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def get_lan_ip() -> str:
    """IP LAN yang dipakai rute keluar. UDP connect tidak mengirim paket apa pun."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        s.close()


def is_local(request: Request) -> bool:
    return request.client is not None and request.client.host in LOCAL_HOSTS


def read_page(name: str) -> HTMLResponse:
    return HTMLResponse((WEB_DIR / name).read_text(encoding="utf-8"), headers=NO_STORE)


def create_app(session: Session, manager: ClientManager, click_test: ClickTest,
               engine: AudioEngine | None, ip: str, port: int) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        watcher = None
        if engine is not None:
            try:
                engine.start(asyncio.get_running_loop())
                print(f"  Audio     : {engine.describe()}", flush=True)
                watcher = asyncio.create_task(engine.watch())
            except Exception as e:
                print(f"  ! Audio PC tidak bisa ditangkap: {e}", flush=True)
        yield
        if watcher is not None:
            watcher.cancel()
        if engine is not None:
            engine.stop()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    @app.get("/")
    async def client_page():
        return read_page("client.html")

    @app.get("/api/join")
    async def join(request: Request, code: str = ""):
        who = request.client.host if request.client else "?"
        status, value = session.try_code(who, code.strip())
        if status == "ok":
            return JSONResponse({"token": value}, headers=NO_STORE)
        if status == "locked":
            return JSONResponse({"error": "terlalu_banyak_percobaan", "retry_after": value},
                                status_code=429, headers=NO_STORE)
        return JSONResponse({"error": "kode_salah"}, status_code=403, headers=NO_STORE)

    @app.get("/host")
    async def host_page(request: Request):
        if not is_local(request):
            return PlainTextResponse("Dashboard hanya bisa dibuka dari PC host.", status_code=403)
        return read_page("host.html")

    @app.get("/api/host/state")
    async def host_state(request: Request):
        if not is_local(request):
            return PlainTextResponse("Hanya dari PC host.", status_code=403)
        return JSONResponse({
            "url": f"http://{ip}:{port}/?t={session.token}",
            "short": f"{ip}:{port}",
            "code": session.code,
            "clients": manager.snapshot(),
        }, headers=NO_STORE)

    @app.post("/api/host/test")
    async def host_test(request: Request):
        if not is_local(request):
            return PlainTextResponse("Hanya dari PC host.", status_code=403)
        return JSONResponse({"started": click_test.start()}, headers=NO_STORE)

    app.include_router(make_ws_router(session, manager, click_test))
    return app


def print_banner(ip: str, port: int, session: Session) -> None:
    join_url = f"http://{ip}:{port}/?t={session.token}"
    qr = qrcode.QRCode(border=1, error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(join_url)
    qr.make(fit=True)
    print()
    print("  SatuSuara  (M3: siaran audio PC)")
    print()
    qr.print_ascii(invert=True)
    print()
    print("  HP        : scan QR di atas, lalu tap Gabung")
    print(f"  Tautan    : {join_url}")
    print(f"  Laptop    : buka {ip}:{port} di Chrome/Edge, masukkan kode")
    print(f"  Kode gabung: {session.code}")
    print(f"  Dashboard : http://localhost:{port}/host  (hanya di PC ini)")
    print()
    print("  Jika penerima tidak bisa membuka halaman:")
    print("   - Firewall Windows: izinkan di Private network")
    print("   - Wi-Fi publik/guest sering memblokir antarperangkat: pakai hotspot HP")
    print()
    print("  Catatan M3: speaker PC ini belum ikut tertunda (M4). Pakai headphone atau")
    print("  kecilkan volume PC saat uji, supaya tidak terdengar gema.")
    print()
    sys.stdout.flush()


def pick_source(args):
    if args.no_audio:
        return None
    if args.tone:
        return ToneSource
    if sys.platform == "win32":
        return LoopbackSource
    print("  ! Capture audio PC hanya untuk Windows. Pakai --tone untuk nada uji.", flush=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(prog="SatuSuara")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ip", help="paksa IP LAN tertentu (jika deteksi otomatis salah, mis. ada VPN)")
    parser.add_argument("--tone", action="store_true",
                        help="siarkan beep uji tiap detik, bukan audio PC")
    parser.add_argument("--no-audio", action="store_true", help="tanpa siaran audio (hanya Tes Bunyi)")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ip = args.ip or get_lan_ip()
    session = Session()
    manager = ClientManager()
    click_test = ClickTest(manager)
    source_cls = pick_source(args)
    engine = AudioEngine(manager, source_cls, DEFAULT_DELAY_MS) if source_cls else None
    app = create_app(session, manager, click_test, engine, ip, args.port)
    print_banner(ip, args.port, session)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
