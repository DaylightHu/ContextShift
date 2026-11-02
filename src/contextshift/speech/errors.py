"""Errors raised by the optional speech integration."""


class SpeechConfigurationError(RuntimeError):
    """The selected model or audio configuration is invalid."""


class SpeechDependencyError(RuntimeError):
    """An optional local speech dependency is not installed."""
