"""Jam server dan Tes Bunyi (M1)."""
import asyncio
import time


def now_ms() -> float:
    # perf_counter, bukan monotonic: di Windows (Python < 3.13) time.monotonic()
    # beresolusi ~15,6 ms, terlalu kasar untuk target sinkron 30 ms.
    return time.perf_counter() * 1000.0


class ClickTest:
    """Menjadwalkan deret klik serempak. Tiap klik dikirim ~lead_ms sebelum waktu putar,
    meniru perilaku streaming audio nanti (M3)."""

    def __init__(self, manager) -> None:
        self.manager = manager
        self._task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, count: int = 10, interval_ms: int = 1000, lead_ms: int = 800) -> bool:
        if self.running:
            return False
        self._task = asyncio.create_task(self._run(count, interval_ms, lead_ms))
        return True

    async def _run(self, count: int, interval_ms: int, lead_ms: int) -> None:
        first = now_ms() + 2 * lead_ms
        for i in range(count):
            play_at = first + i * interval_ms
            wait = (play_at - lead_ms - now_ms()) / 1000.0
            if wait > 0:
                await asyncio.sleep(wait)
            await self.manager.broadcast_json(
                {"type": "click", "play_at_ms": play_at, "seq": i + 1, "total": count}
            )
