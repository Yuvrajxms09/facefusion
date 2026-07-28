import threading
from contextlib import contextmanager
from time import perf_counter
from typing import Dict, Iterator

from facefusion import logger

_LOCK = threading.Lock()
_METRICS: Dict[str, float] = {
    'frames': 0.0,
    'detector_ms': 0.0,
    'landmarker_ms': 0.0,
    'recognizer_ms': 0.0,
    'classifier_ms': 0.0,
    'swapper_onnx_ms': 0.0,
    'swapper_paste_ms': 0.0,
    'swapper_seq_ms': 0.0,
    'frame_total_ms': 0.0,
    'frame_copy_ms': 0.0,
    'audio_ms': 0.0,
    'processor_total_ms': 0.0,
    'video_io_ms': 0.0,
    'enhancer_ms': 0.0,
}


@contextmanager
def measure(name: str, *, log: bool = False) -> Iterator[None]:
    """Accumulate wall-clock time for a named stage.

    The measurement is intentionally host-side. It reflects the time visible
    to the pipeline, including provider synchronization, without forcing an
    extra CUDA synchronization on every frame.
    """
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1000.0
        add(name, elapsed_ms)
        if log:
            logger.debug(f'[profiler] stage={name} elapsed_ms={elapsed_ms:.3f}', __name__)

def add(name: str, ms: float) -> None:
    with _LOCK:
        _METRICS[name] = _METRICS.get(name, 0.0) + float(ms)

def inc_frames(n: int = 1) -> None:
    add('frames', float(n))

def get_and_reset() -> Dict[str, float]:
    with _LOCK:
        snapshot = dict(_METRICS)
        for k in _METRICS.keys():
            _METRICS[k] = 0.0
        return snapshot

def log_summary(context: str = 'job') -> None:
    m = get_and_reset()
    frames = max(1.0, m.get('frames', 0.0))
    def pf(key: str) -> float:
        return m.get(key, 0.0)
    processor_details = ' '.join(
        f"{key}={value:.1f} per_frame={value / frames:.2f}"
        for key, value in sorted(m.items())
        if key.startswith('processor_') and key != 'processor_total_ms'
    )
    logger.info(
        f"[profiler] context={context} frames={int(frames)} "
        f"detector_total_ms={pf('detector_ms'):.1f} per_frame={pf('detector_ms')/frames:.2f} "
        f"landmarker_total_ms={pf('landmarker_ms'):.1f} per_frame={pf('landmarker_ms')/frames:.2f} "
        f"recognizer_total_ms={pf('recognizer_ms'):.1f} per_frame={pf('recognizer_ms')/frames:.2f} "
        f"classifier_total_ms={pf('classifier_ms'):.1f} per_frame={pf('classifier_ms')/frames:.2f} "
        f"swapper_onnx_total_ms={pf('swapper_onnx_ms'):.1f} per_frame={pf('swapper_onnx_ms')/frames:.2f} "
        f"swapper_paste_total_ms={pf('swapper_paste_ms'):.1f} per_frame={pf('swapper_paste_ms')/frames:.2f} "
        f"swapper_seq_total_ms={pf('swapper_seq_ms'):.1f} per_frame={pf('swapper_seq_ms')/frames:.2f} "
        f"enhancer_total_ms={pf('enhancer_ms'):.1f} per_frame={pf('enhancer_ms')/frames:.2f} "
        f"frame_total_ms={pf('frame_total_ms'):.1f} per_frame={pf('frame_total_ms')/frames:.2f} "
        f"frame_copy_ms={pf('frame_copy_ms'):.1f} per_frame={pf('frame_copy_ms')/frames:.2f} "
        f"audio_ms={pf('audio_ms'):.1f} per_frame={pf('audio_ms')/frames:.2f} "
        f"processor_total_ms={pf('processor_total_ms'):.1f} per_frame={pf('processor_total_ms')/frames:.2f} "
        f"video_io_ms={pf('video_io_ms'):.1f} per_frame={pf('video_io_ms')/frames:.2f} "
        f"{processor_details}",
        __name__
    )
