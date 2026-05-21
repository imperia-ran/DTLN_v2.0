"""Project-specific exceptions."""


class MoneyprintError(Exception):
    """Base exception for this package."""


class ConfigurationError(MoneyprintError):
    """Raised when a configuration file is invalid."""


class DatasetError(MoneyprintError):
    """Raised when a dataset structure cannot be interpreted safely."""


class TensorFlowUnavailableError(MoneyprintError):
    """Raised when a TensorFlow-backed path is used without TensorFlow."""
