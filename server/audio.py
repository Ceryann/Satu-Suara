"""Audio PC -> chunk 40 ms (M3): WASAPI loopback, nada uji, dan penyiar chunk."""
import asyncio
import struct
import threading
import time

import numpy as np

from .sync import now_ms

RATE = 48000
CHANNELS = 2
CHUNK_MS = 40
CHUNK_FRAMES = RATE * CHUNK_MS // 1000      # 1920 frame = 7.680 byte (stereo, 16-bit)
REANCHOR_MS = 50                            # selisih jam yang memicu jangkar waktu baru
HEADER = struct.Struct("<BId")              # tipe u8, seq u32, play_at_ms f64 = 13 byte
AUDIO_TYPE = 1


# ---------------------------------------------------------------- konversi format
def to_stereo_f32(raw: bytes, channels: int) -> np.ndarray:
    """PCM int16 interleaved (n kanal) -> float32 stereo (frame, 2)."""
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    x = x.reshape(-1, channels)
    if channels == 2:
        return x
    if channels == 1:
        return np.repeat(x, 2, axis=1)
    if channels in (6, 8):                  # 5.1 / 7.1: FL FR FC LFE BL BR [SL SR]
        k = 0.7071
        left = x[:, 0] + k * x[:, 2] + k * x[:, 4]
        right = x[:, 1] + k * x[:, 2] + k * x[:, 5]
        if channels == 8:
            left = left + k * x[:, 6]
            right = right + k * x[:, 7]
        return np.stack([left, right], axis=1) * 0.6
    return x[:, :2]


def to_pcm16(x: np.ndarray) -> bytes:
    return np.clip(x * 32767.0, -32768, 32767).astype("<i2").tobytes()


def volume_to_gain(volume: float, muted: bool = False) -> float:
    """Skala 0..100 -> gain kuadrat (lebih sesuai pendengaran daripada linear)."""
    return 0.0 if muted else (max(0.0, min(100.0, float(volume))) / 100.0) ** 2


def apply_gain(pcm: bytes, gain: float) -> bytes:
    if gain >= 0.999:
        return pcm
    if gain <= 0.0:
        return bytes(len(pcm))
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) * gain
    return np.clip(x, -32768, 32767).astype("<i2").tobytes()


class LinearResampler:
    """Resampler linear streaming (menjaga fase antarblok). Cukup untuk MVP; Opus/filter lebih baik nanti."""

    def __init__(self, in_rate: int, out_rate: int) -> None:
        self.step = in_rate / out_rate
        self.reset()

    def reset(self) -> None:
        self.pos = 0.0
        self.buf = np.zeros((0, 2), dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self.buf, x]) if len(self.buf) else x
        n = len(buf)
        if n - 1 <= self.pos:
            self.buf = buf
            return np.zeros((0, 2), dtype=np.float32)
        count = int(np.ceil((n - 1 - self.pos) / self.step))
        p = self.pos + np.arange(count) * self.step
        i0 = p.astype(np.int64)
        frac = (p - i0).astype(np.float32)[:, None]
        out = buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac
        nxt = self.pos + count * self.step
        drop = int(nxt)
        self.buf = buf[drop:]
        self.pos = nxt - drop
        return out


class Pipeline:
    """Menerima blok PCM dari perangkat, menghasilkan chunk 40 ms 48 kHz stereo bercap waktu capture."""

    def __init__(self, in_rate: int, in_channels: int, on_chunk) -> None:
        self.in_rate = in_rate
        self.in_channels = in_channels
        self.on_chunk = on_chunk            # on_chunk(capture_ms, pcm_bytes)
        self.resampler = None if in_rate == RATE else LinearResampler(in_rate, RATE)
        self._clear()

    def _clear(self) -> None:
        self.anchor = None                  # waktu capture (ms) frame output ke-0
        self.emitted = 0                    # frame output terkirim sejak jangkar
        self.pending = np.zeros((0, 2), dtype=np.float32)
        if self.resampler is not None:
            self.resampler.reset()

    def feed(self, raw: bytes, t_end_ms: float) -> None:
        if not raw:
            return
        x = to_stereo_f32(raw, self.in_channels)
        t_start = t_end_ms - len(x) * 1000.0 / self.in_rate
        if self.anchor is not None:
            expected = self.anchor + (self.emitted + len(self.pending)) * 1000.0 / RATE
            if abs(t_start - expected) > REANCHOR_MS:   # mis. setelah hening: loopback berhenti mengirim
                self._clear()
        if self.anchor is None:
            self.anchor = t_start
        y = x if self.resampler is None else self.resampler.process(x)
        self.pending = np.concatenate([self.pending, y]) if len(self.pending) else y
        while len(self.pending) >= CHUNK_FRAMES:
            chunk, self.pending = self.pending[:CHUNK_FRAMES], self.pending[CHUNK_FRAMES:]
            capture_ms = self.anchor + self.emitted * 1000.0 / RATE
            self.emitted += CHUNK_FRAMES
            self.on_chunk(capture_ms, to_pcm16(chunk))


# ---------------------------------------------------------------- sumber audio
class LoopbackSource:
    """WASAPI loopback pada output default Windows (mode Sumber saja, S-04)."""

    def __init__(self, on_chunk) -> None:
        self.on_chunk = on_chunk
        self.description = ""
        self._p = None
        self._stream = None

    def start(self) -> None:
        import pyaudiowpatch as pyaudio     # hanya ada di Windows

        p = pyaudio.PyAudio()
        try:
            dev = self._find_loopback(p, pyaudio)
            rate = int(dev["defaultSampleRate"])
            channels = int(dev["maxInputChannels"])
            pipe = Pipeline(rate, channels, self.on_chunk)
            failed = []

            def callback(in_data, frame_count, time_info, status):
                try:
                    pipe.feed(in_data, now_ms())
                except Exception as e:       # jangan matikan stream; lapor sekali
                    if not failed:
                        failed.append(1)
                        print(f"  ! Kesalahan olah audio: {e!r}", flush=True)
                return (None, pyaudio.paContinue)

            self._stream = p.open(
                format=pyaudio.paInt16, channels=channels, rate=rate,
                frames_per_buffer=max(128, rate // 100), input=True,
                input_device_index=dev["index"], stream_callback=callback,
            )
        except Exception:
            p.terminate()
            raise
        self._p = p
        self.description = f"loopback '{dev['name']}', {rate} Hz, {channels} kanal -> 48000 Hz stereo"

    @staticmethod
    def _find_loopback(p, pyaudio):
        if hasattr(p, "get_default_wasapi_loopback"):
            return p.get_default_wasapi_loopback()
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        spk = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if spk.get("isLoopbackDevice"):
            return spk
        for lb in p.get_loopback_device_info_generator():
            if spk["name"] in lb["name"]:
                return lb
        raise RuntimeError("loopback untuk output default tidak ditemukan")

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


class ToneSource:
    """Beep 1 kHz tiap detik untuk uji sinkron tanpa memutar apa pun di PC (--tone)."""

    def __init__(self, on_chunk) -> None:
        self.on_chunk = on_chunk
        self.description = "nada uji: beep 1 kHz tiap detik (bukan audio PC)"
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="tone", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _render(self, n: int) -> bytes:
        idx = (n * CHUNK_FRAMES + np.arange(CHUNK_FRAMES)) % RATE
        t = idx / RATE
        beep = 0.06
        fade = 0.005
        env = np.clip(np.minimum(t, beep - t) / fade, 0.0, 1.0) * (t < beep)
        mono = (np.sin(2 * np.pi * 1000.0 * t) * env * 0.6).astype(np.float32)
        return to_pcm16(np.stack([mono, mono], axis=1))

    def _run(self) -> None:
        t0 = now_ms() + 100.0
        n = 0
        while not self._stop.is_set():
            capture_ms = t0 + n * CHUNK_MS
            wait = (capture_ms + CHUNK_MS - now_ms()) / 1000.0   # chunk "selesai ditangkap"
            if wait > 0 and self._stop.wait(wait):
                break
            self.on_chunk(capture_ms, self._render(n))
            n += 1


# ---------------------------------------------------------------- penyiar
class AudioEngine:
    """Memberi seq dan play_at pada tiap chunk lalu menyiarkannya ke semua penerima (S-05, S-06, S-13)."""

    def __init__(self, manager, source_cls, delay_ms: int = 500, host_player=None) -> None:
        self.manager = manager
        self.source_cls = source_cls
        self.delay_ms = delay_ms
        self.host_player = host_player      # M4: speaker PC ikut memutar pada play_at yang sama
        self.host_as_client = False         # True: PC host tidak memutar sendiri, browser-nya jadi penerima
        self.master_volume = 100            # volume utama untuk SEMUA perangkat (0..100)
        self.master_muted = False
        self.seq = 0
        self.source = None
        self._loop = None
        self._last_chunk = 0.0
        self._warned = False

    def start(self, loop) -> None:
        self._loop = loop
        if self.host_player is not None:
            try:
                self.host_player.start()
            except Exception as e:
                print(f"  ! Speaker PC tidak bisa dibuka: {e}", flush=True)
                self.host_player = None
        self.source = self.source_cls(self._from_thread)
        self.source.start()

    def set_master(self, volume=None, muted=None) -> None:
        if volume is not None:
            self.master_volume = int(max(0, min(100, volume)))
        if muted is not None:
            self.master_muted = bool(muted)

    def status(self) -> dict:
        age = (time.monotonic() - self._last_chunk) if self._last_chunk else None
        return {
            "capture": getattr(self.source, "description", "") if self.source is not None else "",
            "audio_active": age is not None and age < 1.5,
        }

    def describe(self) -> str:
        text = getattr(self.source, "description", "")
        if self.host_player is not None:
            text += f"\n  Speaker PC: {self.host_player.description} (ikut bersuara, delay {self.delay_ms} ms)"
        return text

    def stop(self) -> None:
        if self.source is not None:
            self.source.stop()
        if self.host_player is not None:
            self.host_player.stop()

    def _from_thread(self, capture_ms: float, pcm: bytes) -> None:
        self._loop.call_soon_threadsafe(self._broadcast, capture_ms, pcm)

    def _broadcast(self, capture_ms: float, pcm: bytes) -> None:
        self._last_chunk = time.monotonic()
        play_at = capture_ms + self.delay_ms
        pcm = apply_gain(pcm, volume_to_gain(self.master_volume, self.master_muted))
        if self.host_player is not None:    # speaker PC selalu ikut, walau belum ada HP
            self.host_player.push(play_at, pcm)
        if self.manager.count() == 0:       # S-13: tidak ada penerima, tidak mengirim
            return
        data = HEADER.pack(AUDIO_TYPE, self.seq & 0xFFFFFFFF, play_at) + pcm
        self.seq += 1
        self.manager.broadcast_audio(data)

    async def watch(self) -> None:
        """Loopback tidak mengirim data saat PC hening; beri tahu pengguna sekali."""
        while True:
            await asyncio.sleep(2)
            idle = time.monotonic() - self._last_chunk > 3
            if idle and self.manager.count() > 0 and not self._warned:
                print("  ! Belum ada audio dari PC. Putar sesuatu di PC host (YouTube, Spotify, dll).", flush=True)
                self._warned = True
            elif not idle:
                self._warned = False
