"""SatuSuara M0: kerangka server, token, kode gabung, QR di terminal."""
import argparse
import socket
import sys
from pathlib import Path

import qrcode
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .session import Session
from .ws import ClientManager, make_ws_router

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


def create_app(session: Session, manager: ClientManager, ip: str, port: int) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

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

    app.include_router(make_ws_router(session, manager))
    return app


def print_banner(ip: str, port: int, session: Session) -> None:
    join_url = f"http://{ip}:{port}/?t={session.token}"
    qr = qrcode.QRCode(border=1, error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(join_url)
    qr.make(fit=True)
    print()
    print("  SatuSuara  (M0: kerangka server)")
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
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(prog="SatuSuara")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ip", help="paksa IP LAN tertentu (jika deteksi otomatis salah, mis. ada VPN)")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ip = args.ip or get_lan_ip()
    session = Session()
    manager = ClientManager()
    app = create_app(session, manager, ip, args.port)
    print_banner(ip, args.port, session)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
