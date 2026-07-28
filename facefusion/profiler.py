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

_WALL_METRIC_PREFIXES = ('wall_', 'model_load_')


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

def reset() -> None:
    with _LOCK:
        for key in _METRICS:
            _METRICS[key] = 0.0

def log_summary(context: str = 'job') -> None:
    m = get_and_reset()
    frames = max(1.0, m.get('frames', 0.0))
    accumulated_details = []
    wall_details = []

    for key, value in sorted(m.items()):
        if key == 'frames' or value == 0:
            continue
        metric = key.removesuffix('_ms')
        if key.startswith(_WALL_METRIC_PREFIXES):
            wall_details.append(f'{metric}={value:.1f}')
        else:
            accumulated_details.append(f'{metric}={value:.1f} per_frame={value / frames:.2f}')

    logger.info(
        f"[profiler] context={context} kind=wall_ms {' '.join(wall_details)}",
        __name__
    )
    logger.info(
        f"[profiler] context={context} kind=accumulated_worker_ms frames={int(frames)} "
        f"{' '.join(accumulated_details)}",
        __name__
    )
