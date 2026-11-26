"""SatuSuara: server, token, kode gabung, QR, jam server, Tes Bunyi, siaran audio PC."""
import argparse
import asyncio
import math
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import qrcode
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .audio import HOST_MAX_VOLUME, MAX_DELAY_MS, MIN_DELAY_MS, AudioEngine, LoopbackSource, ToneSource
from .player import HOST_BUFFER_MS, HostPlayer, HostPlan, load_config, plan_host, print_devices, save_config
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


def mode_label(engine: AudioEngine | None) -> str:
    if engine is None:
        return "Tanpa audio (hanya Tes Bunyi)"
    tone = isinstance(engine.source, ToneSource)
    play = engine.host_player is not None
    if tone:
        return "Nada uji, speaker PC ikut memutar" if play else "Nada uji, speaker PC TIDAK ikut (tanpa --host-play)"
    if play:
        return "PC ikut bersuara (tersinkron)"
    if engine.host_as_client:
        return "Host sebagai klien: speaker PC diputar lewat browser di PC ini"
    return "Sumber saja: speaker PC memutar langsung tanpa delay"


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
        host_play = engine is not None and engine.host_player is not None
        return JSONResponse({
            "url": f"http://{ip}:{port}/?t={session.token}",
            "short": f"{ip}:{port}",
            "code": session.code,
            "clients": manager.snapshot(),
            "mode": mode_label(engine),
            "host_play": host_play,
            "host_offset_ms": engine.host_player.offset_ms if host_play else None,
            "speaker": engine.host_player.description if host_play else None,
            "host_stats": engine.host_player.stats() if host_play else None,
            "delay_ms": engine.delay_ms if engine is not None else None,
            "delay_limits": [MIN_DELAY_MS, MAX_DELAY_MS],
            **(engine.status() if engine is not None else {"capture": "", "audio_active": False}),
            "master": ({"volume": engine.master_volume, "muted": engine.master_muted}
                       if engine is not None else None),
            "host_volume": ({"volume": engine.host_player.volume, "muted": engine.host_player.muted}
                            if host_play else None),
        }, headers=NO_STORE)

    @app.post("/api/host/offset")
    async def host_offset(request: Request):
        if not is_local(request):
            return PlainTextResponse("Hanya dari PC host.", status_code=403)
        player = engine.host_player if engine is not None else None
        if player is None:
            return JSONResponse({"error": "tidak_ada_pemutar_host"}, status_code=409, headers=NO_STORE)
        try:
            value = float((await request.json())["offset_ms"])
            if not math.isfinite(value):
                raise ValueError
        except Exception:
            return JSONResponse({"error": "offset_ms_tidak_valid"}, status_code=400, headers=NO_STORE)
        value = max(-500.0, min(500.0, value))
        player.set_offset(value)
        save_config({**load_config(), "host_offset_ms": value})   # diingat untuk lain kali
        return JSONResponse({"host_offset_ms": value}, headers=NO_STORE)

    @app.post("/api/host/delay")
    async def host_delay(request: Request):
        """Delay target (ms) untuk semua perangkat, termasuk speaker PC."""
        if not is_local(request):
            return PlainTextResponse("Hanya dari PC host.", status_code=403)
        if engine is None:
            return JSONResponse({"error": "tanpa_audio"}, status_code=409, headers=NO_STORE)
        try:
            value = float((await request.json())["delay_ms"])
            if not math.isfinite(value):
                raise ValueError
        except Exception:
            return JSONResponse({"error": "delay_ms_tidak_valid"}, status_code=400, headers=NO_STORE)
        applied = engine.set_delay(value)
        save_config({**load_config(), "delay_ms": applied})       # diingat untuk lain kali
        return JSONResponse({"delay_ms": applied}, headers=NO_STORE)

    @app.post("/api/host/control")
    async def host_control(request: Request):
        """Volume/mute: target 'master' (semua), 'host' (speaker PC), atau 'client' (+id)."""
        if not is_local(request):
            return PlainTextResponse("Hanya dari PC host.", status_code=403)

        def bad(msg: str, code: int = 400):
            return JSONResponse({"error": msg}, status_code=code, headers=NO_STORE)

        try:
            data = await request.json()
            target = data.get("target")
            volume = data.get("volume")
            muted = data.get("muted")
            if volume is not None:
                if isinstance(volume, bool) or not isinstance(volume, (int, float)) or not math.isfinite(volume):
                    return bad("volume_tidak_valid")
                volume = int(max(0, min(HOST_MAX_VOLUME, round(volume))))    # >100 hanya berlaku untuk speaker PC
            if muted is not None and not isinstance(muted, bool):
                return bad("muted_tidak_valid")
        except Exception:
            return bad("json_tidak_valid")
        if volume is None and muted is None:
            return bad("volume_atau_muted_diperlukan")

        if target == "master":
            if engine is None:
                return bad("tidak_ada_audio", 409)
            engine.set_master(volume, muted)
            if volume is not None:
                save_config({**load_config(), "master_volume": engine.master_volume})
            return JSONResponse({"volume": engine.master_volume, "muted": engine.master_muted}, headers=NO_STORE)
        if target == "host":
            hp = engine.host_player if engine is not None else None
            if hp is None:
                return bad("tidak_ada_pemutar_host", 409)
            hp.set_volume(volume, muted)
            if volume is not None:
                save_config({**load_config(), "host_volume": hp.volume})
            return JSONResponse({"volume": hp.volume, "muted": hp.muted}, headers=NO_STORE)
        if target == "client":
            try:
                cid = int(data.get("id"))
            except (TypeError, ValueError):
                return bad("id_tidak_valid")
            if not manager.set_control(cid, volume, muted):
                return bad("klien_tidak_ditemukan", 404)
            return JSONResponse({"ok": True}, headers=NO_STORE)
        return bad("target_tidak_valid")

    @app.post("/api/host/disconnect")
    async def host_disconnect(request: Request):
        """Putuskan satu perangkat ({"id": n}) atau semuanya ({"all": true})."""
        if not is_local(request):
            return PlainTextResponse("Hanya dari PC host.", status_code=403)
        try:
            data = await request.json()
            if not isinstance(data, dict):
                raise ValueError
        except Exception:
            return JSONResponse({"error": "json_tidak_valid"}, status_code=400, headers=NO_STORE)
        if data.get("all") is True:
            names = [c["name"] for c in manager.snapshot()]
            n = manager.kick_all()
            for name in names:
                print(f"  x {name} diputus oleh host.", flush=True)
            return JSONResponse({"disconnected": n}, headers=NO_STORE)
        try:
            cid = int(data.get("id"))
        except (TypeError, ValueError):
            return JSONResponse({"error": "id_tidak_valid"}, status_code=400, headers=NO_STORE)
        name = next((c["name"] for c in manager.snapshot() if c["id"] == cid), None)
        if not manager.kick(cid):
            return JSONResponse({"error": "klien_tidak_ditemukan"}, status_code=404, headers=NO_STORE)
        print(f"  x {name} diputus oleh host.", flush=True)
        return JSONResponse({"disconnected": 1}, headers=NO_STORE)

    @app.post("/api/host/test")
    async def host_test(request: Request):
        if not is_local(request):
            return PlainTextResponse("Hanya dari PC host.", status_code=403)
        return JSONResponse({"started": click_test.start()}, headers=NO_STORE)

    app.include_router(make_ws_router(session, manager, click_test))
    return app


def print_banner(ip: str, port: int, session: Session, plan: HostPlan | None = None) -> None:
    join_url = f"http://{ip}:{port}/?t={session.token}"
    qr = qrcode.QRCode(border=1, error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(join_url)
    qr.make(fit=True)
    print()
    print("  SatuSuara")
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
    if plan is not None:
        for line in plan.lines:
            print(f"  {line}")
        if plan.lines:
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
    parser.add_argument("--host-play", action="store_true",
                        help="dengan --tone: speaker PC ikut memutar beep (uji sinkron tanpa perangkat virtual)")
    parser.add_argument("--no-host-play", action="store_true",
                        help="PC host tidak memutar sendiri; pakai browser di PC ini sebagai penerima biasa")
    parser.add_argument("--speaker", help="sebagian nama speaker fisik untuk diputar PC host")
    parser.add_argument("--host-offset", type=float, default=None,
                        help="offset speaker PC dalam ms (positif = lebih lambat); default: nilai terakhir "
                             "dari dashboard")
    parser.add_argument("--host-buffer", type=float, default=None,
                        help=f"buffer speaker PC dalam ms (default {HOST_BUFFER_MS:g}; naikkan, mis. 40, jika suara "
                             "speaker PC berkresek)")
    parser.add_argument("--delay", type=int, default=None,
                        help=f"delay target semua perangkat dalam ms ({MIN_DELAY_MS}-{MAX_DELAY_MS}); default: nilai "
                             "terakhir dari dashboard, awalnya 500")
    parser.add_argument("--list-devices", action="store_true", help="tampilkan perangkat output lalu keluar")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.list_devices:
        print_devices()
        return

    ip = args.ip or get_lan_ip()
    session = Session()
    manager = ClientManager()
    click_test = ClickTest(manager)
    source_cls = pick_source(args)
    plan = None
    host_player = None
    if source_cls is LoopbackSource or (source_cls is ToneSource and args.host_play):
        plan = plan_host(source_cls is LoopbackSource, args.host_play, args.speaker, args.no_host_play)
        if plan.host_play:
            offset = args.host_offset
            if offset is None:
                offset = float(load_config().get("host_offset_ms", 0.0))
            buf = args.host_buffer
            if buf is None:
                buf = float(load_config().get("host_buffer_ms", HOST_BUFFER_MS))
            host_player = HostPlayer(plan.speaker_name, offset_ms=offset, buffer_ms=buf)
    delay = args.delay if args.delay is not None else load_config().get("delay_ms", DEFAULT_DELAY_MS)
    try:
        delay = int(delay)
    except (TypeError, ValueError):
        delay = DEFAULT_DELAY_MS
    engine = AudioEngine(manager, source_cls, delay, host_player) if source_cls else None
    if engine is not None:
        cfg = load_config()
        engine.set_master(cfg.get("master_volume", 100))        # mute tidak diingat, volume ya
        if host_player is not None:
            host_player.set_volume(cfg.get("host_volume", 100))
    if engine is not None and plan is not None:
        engine.host_as_client = args.no_host_play and plan.default_virtual
    app = create_app(session, manager, click_test, engine, ip, args.port)
    print_banner(ip, args.port, session, plan)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
