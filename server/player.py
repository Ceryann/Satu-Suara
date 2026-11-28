"""Pemutar speaker host (M4): memutar audio pada play_at yang sama dengan penerima (WASAPI)."""
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import HOST_MAX_VOLUME, RATE, LinearResampler, to_stereo_f32, volume_to_gain

TOL_MS = 3.0            # selisih jam yang dibiarkan; lebih dari ini antrean dikoreksi (drop/diam)
MAX_QUEUE_S = 3
HOST_BUFFER_MS = 20     # ukuran buffer callback WASAPI; lebih besar = lebih tahan GIL/jitter Python (kresek)
SLIP_ON_MS = 0.3       # selisih (ms) yang mulai dikoreksi halus (merentang 1 frame per callback)
SLIP_OFF_MS = 0.1       # berhenti mengoreksi bila selisih sudah di bawah ini (histeresis)
OFF_WINDOW_S = 1.0      # lama riwayat (dtk) untuk menaksir waktu DAC yang stabil
PA_OUTPUT_UNDERFLOW = 0x04
VIRTUAL_HINTS = ("cable", "vb-audio", "virtual", "voicemeeter")
GOOD_HINTS = ("speaker", "headphone", "earphone", "realtek", "headset")
BAD_HINTS = ("hdmi", "display", "digital", "spdif", "monitor", "nvidia", "amd")


def soft_clip(x, knee: float = 0.7):
    """Pembatas lembut untuk gain > 1: sampel di bawah 'knee' tak berubah, di atasnya dipadatkan
    mulus mendekati 1,0 (tanpa pemotongan keras yang terdengar pecah)."""
    ax = np.abs(x)
    over = ax > knee
    if not over.any():
        return x
    y = x.copy()
    r = 1.0 - knee
    y[over] = np.sign(x[over]) * (knee + r * np.tanh((ax[over] - knee) / r))
    return y


# ---------------------------------------------------------------- konfigurasi kecil
def config_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "SatuSuara" / "config.json"


def load_config() -> dict:
    try:
        return json.loads(config_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    try:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------- perangkat
def is_virtual(name: str) -> bool:
    n = (name or "").lower()
    return any(h in n for h in VIRTUAL_HINTS)


def list_wasapi_outputs(p, pyaudio):
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    outs = []
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d.get("hostApi") == wasapi["index"] and d.get("maxOutputChannels", 0) > 0:
            outs.append(d)
    default = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    return outs, default


def pick_speaker(outs, hint=None, saved=None):
    """Pilih speaker fisik (bukan perangkat virtual) untuk diputar oleh host."""
    real = [d for d in outs if not is_virtual(d["name"])]
    if hint:
        match = [d for d in real if hint.lower() in d["name"].lower()]
        if not match:
            raise RuntimeError(f"speaker '{hint}' tidak ditemukan (coba: python run.py --list-devices)")
        return match[0]
    if saved:
        for d in real:
            if d["name"] == saved:
                return d
    if not real:
        raise RuntimeError("tidak ada speaker fisik selain perangkat virtual")

    def score(d):
        n = d["name"].lower()
        return sum(h in n for h in GOOD_HINTS) - sum(h in n for h in BAD_HINTS)

    return max(real, key=score)


def print_devices() -> None:
    if sys.platform != "win32":
        print("  --list-devices hanya untuk Windows.")
        return
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    try:
        outs, default = list_wasapi_outputs(p, pyaudio)
        print("\n  Perangkat output WASAPI:")
        for d in outs:
            tag = []
            if d["index"] == default["index"]:
                tag.append("DEFAULT Windows")
            if is_virtual(d["name"]):
                tag.append("virtual")
            print(f"   - {d['name']}  ({int(d['defaultSampleRate'])} Hz, {d['maxOutputChannels']} kanal)"
                  + (f"  [{', '.join(tag)}]" if tag else ""))
        print("\n  Pilih speaker tertentu dengan:  python run.py --speaker \"sebagian nama\"\n")
    finally:
        p.terminate()


@dataclass
class HostPlan:
    host_play: bool = False
    default_virtual: bool = False
    speaker_name: str | None = None
    lines: list = field(default_factory=list)      # pesan untuk pengguna


def plan_host(use_loopback: bool, force_host_play: bool, speaker_hint: str | None,
              host_as_client: bool = False) -> HostPlan:
    """Tentukan mode: 'PC ikut bersuara' (output default = virtual) atau 'Sumber saja'."""
    plan = HostPlan()
    if sys.platform != "win32":
        if force_host_play:
            plan.lines.append("Pemutar speaker host hanya untuk Windows.")
        return plan
    try:
        import pyaudiowpatch as pyaudio
        p = pyaudio.PyAudio()
    except Exception as e:
        plan.lines.append(f"Tidak bisa membaca perangkat audio: {e}")
        return plan
    try:
        outs, default = list_wasapi_outputs(p, pyaudio)
        cfg = load_config()
        default_virtual = is_virtual(default["name"])
        plan.default_virtual = default_virtual
        virtual_present = any(is_virtual(d["name"]) for d in outs)
        if not default_virtual:
            save_config({**cfg, "speaker": default["name"]})   # ingat speaker asli untuk nanti

        try:
            if use_loopback and default_virtual and host_as_client:
                plan.lines += [
                    "Mode      : Host sebagai klien (speaker PC diputar lewat browser di PC ini)",
                    "Langkah   : buka tautan penerima di browser PC ini, lalu tap Gabung. Arahkan browser itu",
                    "            ke speaker asli: Settings > System > Sound > Volume mixer > (browser) >",
                    "            Output device. Putar musik di browser LAIN agar masuk ke CABLE.",
                    "Awas loop : jika terdengar dengung/gema yang makin keras, browser penerima ini masih",
                    "            mengeluarkan suara ke CABLE. Tutup halaman penerima di PC ini.",
                ]
            elif use_loopback and default_virtual:
                spk = pick_speaker(outs, speaker_hint, cfg.get("speaker"))
                plan.host_play, plan.speaker_name = True, spk["name"]
                plan.lines += [
                    "Mode      : PC ikut bersuara (tersinkron dengan HP)",
                    f"Speaker   : {spk['name']}  (ganti: --speaker \"nama\", daftar: --list-devices)",
                    "Perhatian : output default Windows sedang diarahkan ke perangkat virtual.",
                    "            Jika SatuSuara ditutup, suara PC hilang sampai kamu memilih speaker",
                    "            lagi di Settings > System > Sound > Output.",
                ]
            elif use_loopback:
                plan.lines.append("Mode      : Sumber saja (speaker PC memutar langsung, TANPA delay)")
                if virtual_present:
                    plan.lines += [
                        "Agar PC ikut tersinkron: Settings > System > Sound > Output, pilih",
                        "'CABLE Input (VB-Audio Virtual Cable)', lalu jalankan ulang SatuSuara.",
                    ]
                else:
                    plan.lines += [
                        "Agar PC ikut tersinkron perlu 1 perangkat audio virtual (sekali pasang):",
                        "unduh VB-CABLE di vb-audio.com/Cable, jalankan setup sebagai administrator,",
                        "restart PC, lalu pilih 'CABLE Input' sebagai output Windows.",
                    ]
            elif force_host_play:
                saved = default["name"] if not default_virtual else cfg.get("speaker")
                spk = pick_speaker(outs, speaker_hint, saved)
                plan.host_play, plan.speaker_name = True, spk["name"]
                plan.lines.append(f"Mode      : nada uji, PC ikut memutar lewat '{spk['name']}'")
        except RuntimeError as e:
            plan.lines.append(f"! {e}")
    except Exception as e:
        plan.host_play = False
        plan.lines.append(f"! Tidak bisa membaca perangkat audio: {e}")
    finally:
        p.terminate()
    return plan


# ---------------------------------------------------------------- pemutar
class HostPlayer:
    """Antrean linimasa audio yang dikonsumsi callback WASAPI sesuai jam server.

    Tiap callback: waktu (jam server) saat frame pertama buffer mencapai DAC =
    perf_counter sekarang + latensi yang dilaporkan PortAudio - offset manual.
    Antrean diselaraskan ke waktu itu: sedikit selisih dibiarkan, selisih besar dikoreksi.
    """

    def __init__(self, speaker_name: str, offset_ms: float = 0.0, gain: float = 1.0,
                 buffer_ms: float = HOST_BUFFER_MS) -> None:
        self.speaker_name = speaker_name
        self.buffer_ms = max(5.0, min(100.0, float(buffer_ms)))
        self.offset_ms = offset_ms          # positif = putar lebih lambat (sama seperti slider HP)
        self.gain = gain
        self.volume = 100                   # 0..100, diatur dari dashboard
        self.muted = False
        self.description = ""
        self.rate = RATE
        self.ch = 2
        self._p = None
        self._stream = None
        self._rs = None
        self._cont = 0
        self._latency_fallback = 0.05
        self._lock = threading.Lock()
        self._q: deque = deque()
        self._qframes = 0
        self._t0_ms = 0.0                   # waktu (jam server) frame pertama di antrean
        self._aligned = False               # False: selaraskan presisi; True: toleransi hanya untuk drift
        self._epoch = deque(maxlen=100)     # riwayat (perf_counter - current_time PortAudio), ~1 detik
        # Taksiran waktu DAC: tiap callback menghasilkan "waktu stream-sample-0 sampai ke DAC"; jitter
        # masuk callback (GIL, penjadwal Windows) hanya menambah, jadi diambil persentil rendah.
        self._off_hist = deque(maxlen=max(20, int(OFF_WINDOW_S * 1000.0 / self.buffer_ms)))
        self._out_frames = 0                # frame keluaran sejak stream dibuka (termasuk diam)
        # penghitung diagnostik (tampil di dashboard)
        self.n_underflow = 0                # PortAudio melapor callback terlambat (suara putus)
        self.n_correct = 0                  # koreksi drift KASAR (potong/diam, selisih > TOL_MS) setelah antrean selaras
        self.n_slip = 0                     # koreksi drift halus (1 frame direntang/dimampatkan, tak terdengar)
        self._slipping = False
        self.n_late = 0                     # chunk datang terlambat dan dibuang
        self.n_gap = 0                      # chunk hilang, diisi diam
        self.n_starve = 0                   # antrean habis saat memutar (wajar sekali tiap lagu berhenti)
        self._cb_max_ms = 0.0               # durasi callback terlama (meluruh pelan)

    # ---- siklus hidup
    def start(self) -> None:
        import pyaudiowpatch as pyaudio
        p = pyaudio.PyAudio()
        try:
            outs, _ = list_wasapi_outputs(p, pyaudio)
            dev = next((d for d in outs if d["name"] == self.speaker_name), None)
            if dev is None:
                raise RuntimeError(f"speaker '{self.speaker_name}' tidak ditemukan")
            self.rate = int(dev["defaultSampleRate"])
            self.ch = max(1, int(dev["maxOutputChannels"]))
            self._rs = None if self.rate == RATE else LinearResampler(RATE, self.rate)
            self._cont = pyaudio.paContinue
            self._stream = p.open(
                format=pyaudio.paFloat32, channels=self.ch, rate=self.rate, output=True,
                output_device_index=dev["index"], frames_per_buffer=max(128, int(self.rate * self.buffer_ms / 1000.0)),
                stream_callback=self._callback,
            )
            try:
                self._latency_fallback = float(self._stream.get_output_latency())
            except Exception:
                pass
        except Exception:
            p.terminate()
            raise
        self._p = p
        self.description = f"{dev['name']}, {self.rate} Hz, {self.ch} kanal"

    def stop(self) -> None:
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
            if self._p is not None:
                self._p.terminate()
        except Exception:
            pass
        self._stream = self._p = None

    def set_volume(self, volume=None, muted=None) -> None:
        if volume is not None:
            self.volume = int(max(0, min(HOST_MAX_VOLUME, volume)))
        if muted is not None:
            self.muted = bool(muted)
        self.gain = volume_to_gain(self.volume, self.muted, HOST_MAX_VOLUME)

    def set_offset(self, ms: float) -> None:
        """Ubah offset saat berjalan. Selaraskan ulang presisi agar langkah di bawah TOL_MS pun berlaku."""
        with self._lock:
            self.offset_ms = float(ms)
            self._aligned = False

    # ---- sisi pengisi (thread asyncio)
    def push(self, play_at_ms: float, pcm: bytes) -> None:
        x = to_stereo_f32(pcm, 2)
        if self._rs is not None:
            x = self._rs.process(x)
        if not len(x):
            return
        with self._lock:
            if self._qframes == 0:
                self._t0_ms = play_at_ms
                self._aligned = False
            else:
                end = self._t0_ms + self._qframes * 1000.0 / self.rate
                gap = play_at_ms - end
                if gap > TOL_MS / 2:
                    if gap < 2500.0:       # chunk hilang / delay dinaikkan: isi diam agar linimasa tetap benar
                        self.n_gap += 1
                        pad = np.zeros((int(round(gap * self.rate / 1000.0)), 2), dtype=np.float32)
                        self._q.append(pad)
                        self._qframes += len(pad)
                    else:                  # celah besar: mulai ulang antrean
                        self._q.clear()
                        self._qframes = 0
                        self._t0_ms = play_at_ms
                        self._aligned = False
                elif gap < -TOL_MS / 2:    # tumpang tindih / terlambat: buang
                    self.n_late += 1
                    return
            self._q.append(x)
            self._qframes += len(x)
            if self._qframes > MAX_QUEUE_S * self.rate:
                self._q.clear()
                self._qframes = 0
                self._aligned = False

    # ---- sisi pemakai (thread PortAudio)
    def _take(self, n: int):
        parts, got = [], 0
        while got < n and self._q:
            a = self._q[0]
            need = n - got
            if len(a) <= need:
                parts.append(a)
                got += len(a)
                self._q.popleft()
            else:
                parts.append(a[:need])
                self._q[0] = a[need:]
                got += need
        self._qframes -= got
        data = np.concatenate(parts) if parts else np.zeros((0, 2), dtype=np.float32)
        return data, got

    def _estimate_off(self) -> float:
        """Persentil ke-10 dari riwayat: lateness callback hanya menambah, jadi nilai rendah paling jujur."""
        h = sorted(self._off_hist)
        return h[len(h) // 10]

    def stats(self) -> dict:
        try:
            h = sorted(self._off_hist)
        except RuntimeError:                 # deque berubah saat disalin (thread callback)
            h = []
        jitter = (h[int(len(h) * 0.9)] - h[len(h) // 10]) if len(h) >= 10 else 0.0
        return {
            "underflow": self.n_underflow, "correct": self.n_correct, "slip": self.n_slip, "late": self.n_late,
            "gap": self.n_gap, "starve": self.n_starve,
            "jitter_ms": round(jitter, 1), "cb_max_ms": round(self._cb_max_ms, 1),
            "buffer_ms": self.buffer_ms,
        }

    def _callback(self, in_data, frame_count, time_info, status):
        t_now = time.perf_counter()
        if status & PA_OUTPUT_UNDERFLOW:     # callback sebelumnya terlambat: hubungan sampel-waktu patah
            self.n_underflow += 1
            self._off_hist.clear()
        cur = time_info.get("current_time", 0.0)
        dac = time_info.get("output_buffer_dac_time", 0.0)
        if 0.0 <= dac - cur < 1.0:
            # Jadwal DAC dari PortAudio (jam perangkat) dipetakan ke perf_counter. Selisih jam
            # diambil nilai TERKECIL dalam ~1 detik: keterlambatan masuk Python hanya menambah.
            self._epoch.append(t_now - cur)
            t_first_s = dac + min(self._epoch)
        else:                                # PortAudio tidak memberi waktu yang masuk akal
            t_first_s = t_now + self._latency_fallback
        # Waktu sampel-0 stream sampai ke DAC = waktu frame pertama buffer ini dikurangi frame yang sudah
        # keluar. Nilainya tetap; jitter callback ditapis lewat persentil agar tidak memicu koreksi palsu.
        out_ms = self._out_frames * 1000.0 / self.rate
        self._off_hist.append(t_first_s * 1000.0 - out_ms)
        t_first_ms = self._estimate_off() + out_ms - self.offset_ms

        n = frame_count
        self._out_frames += n
        out = np.zeros((n, self.ch), dtype=np.float32)
        with self._lock:
            if self._qframes > 0:
                diff = t_first_ms - self._t0_ms          # >0: antrean terlalu awal
                lead_silence = 0
                slip = 0
                # Selaraskan presisi (level frame) saat mulai/setelah antrean kosong. Sesudahnya: selisih kecil
                # dikoreksi HALUS (rentang/mampatkan buffer sebanyak 1 frame per callback, tak terdengar);
                # hanya lompatan > TOL_MS (mis. setelah buffer underflow) yang dikoreksi kasar.
                if not self._aligned or abs(diff) > TOL_MS:
                    self._slipping = False
                    if self._aligned:
                        self.n_correct += 1
                    if diff > 0:
                        drop = min(self._qframes, int(round(diff * self.rate / 1000.0)))
                        self._take(drop)
                        self._t0_ms += drop * 1000.0 / self.rate
                    elif diff < 0:
                        lead_silence = min(n, int(round(-diff * self.rate / 1000.0)))
                else:
                    if abs(diff) > SLIP_ON_MS:
                        self._slipping = True
                    elif abs(diff) < SLIP_OFF_MS:
                        self._slipping = False
                    if self._slipping:
                        slip = 1 if diff > 0 else -1     # >0: antrean tertinggal -> lahap 1 frame lebih banyak
                if self._qframes > 0 and lead_silence < n:
                    want = n - lead_silence + slip
                    data, got = self._take(want)
                    consumed = got
                    if slip and got == want and got > 1:
                        pos = np.linspace(0.0, got - 1.0, n)        # n frame keluar dari `got` frame masuk
                        idx = np.arange(got, dtype=np.float64)
                        data = np.stack([np.interp(pos, idx, data[:, c]) for c in range(data.shape[1])], axis=1)
                        data = data.astype(np.float32)
                        got = n
                        self.n_slip += 1
                    self._t0_ms += consumed * 1000.0 / self.rate
                    self._aligned = True
                    data = data * self.gain
                    if self.gain > 1.0:
                        data = soft_clip(data)
                    if self.ch == 1:
                        out[lead_silence:lead_silence + got, 0] = data.mean(axis=1)
                    else:
                        out[lead_silence:lead_silence + got, :2] = data
                    if lead_silence + got < n:
                        self.n_starve += 1               # antrean habis di tengah buffer
            else:
                self._aligned = False                    # antrean kosong (underrun): selaraskan ulang nanti
        dur = (time.perf_counter() - t_now) * 1000.0
        self._cb_max_ms = max(dur, self._cb_max_ms * 0.995)
        return (out.tobytes(), self._cont)
