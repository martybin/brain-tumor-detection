"""Explicit failures for data, labels, and I/O. Nothing here is silent."""

from __future__ import annotations


class FrameworkError(Exception):
    """Base error for this research framework."""


class DatasetNotFoundError(FrameworkError):
    """The configured dataset root does not exist."""


class DatasetLayoutError(FrameworkError):
    """Files on disk do not match the dataset schema."""


class MissingModalityError(FrameworkError):
    """A required MRI modality is absent for one or more patients."""


class MissingLabelError(FrameworkError):
    """A task was requested without a confirmed ground-truth label."""


class LabelSchemaError(FrameworkError):
    """Label values are unconfirmed or incompatible with the requested task."""


class RawDataProtectionError(FrameworkError):
    """A write was attempted inside the raw dataset tree."""


class SplitLeakageError(FrameworkError):
    """The same patient appears in more than one split."""


class ConfigurationError(FrameworkError):
    """A config file is missing a required field or names an unknown option."""
