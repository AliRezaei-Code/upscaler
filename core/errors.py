"""Exception hierarchy.

Every failure the app raises on purpose derives from `UpscalerError`, so a
front-end can catch one class and show the message verbatim instead of
matching on text.
"""

from __future__ import annotations


class UpscalerError(Exception):
    """Base class for every error this application raises deliberately."""


class ConfigError(UpscalerError):
    """The job configuration is not runnable as it stands."""


class DeviceProbeError(UpscalerError):
    """A device could not be enumerated or classified at all."""


class ModelFileTooSmall(UpscalerError):
    """The model file is a failed download, not a model."""


class ModelLoadError(UpscalerError):
    """The model file is present but could not be loaded or verified."""


class BackendUnavailableError(UpscalerError):
    """No execution backend can serve this model on this device."""


class DownloadError(UpscalerError):
    """A model or runtime download failed and could not be resumed."""


class UnsupportedScaleError(UpscalerError):
    """The model does not support the requested output scale."""
