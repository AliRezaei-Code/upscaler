"""The single event type every producer and both front-ends speak.

A model download, a community fetch, a runtime install and an upscale job all
report through `PipelineEvent`, distinguished by `kind` and `task_id`. Keeping
one type means one queue, one signal and one log widget for all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EventKind = Literal["stage", "device_progress", "log", "error", "done"]


@dataclass(frozen=True)
class PipelineEvent:
    """One progress or state notification from a producer.

    Attributes:
        kind: What happened. `stage` marks a pipeline step, `device_progress`
            a frame-count update from one device, `log` a line of text, `error`
            a failure, `done` the end of a task.
        task_id: Which producer emitted it — `"job"`, `"model:<id>"`,
            `"community"`, `"runtime:<name>"`. Front-ends dispatch on this, so
            a model download's progress can never land in the job's widgets.
        stage: Pipeline step name for `kind="stage"`.
        device_index: The device's own index, which is unique only within its
            vendor's device space — GPU 0 and CPU 0 are both 0. `-1` when not
            device-specific.
        device_ordinal: Which *selected* device the update came from, 0-based,
            unique across the whole selection. This is what a front-end keys a
            row on; `device_index` is for the log.
        processed: Frames finished so far by the emitting device.
        total: Frames the emitting device was given.
        fps: Measured throughput of the emitting device, frames per second.
        eta_seconds: Projected seconds remaining, `0.0` when not yet known.
        message: Human-readable text for `stage`, `log` and `error`.
    """

    kind: EventKind
    task_id: str = ""
    stage: str = ""
    device_index: int = -1
    device_ordinal: int = -1
    processed: int = 0
    total: int = 0
    fps: float = 0.0
    eta_seconds: float = 0.0
    message: str = ""
