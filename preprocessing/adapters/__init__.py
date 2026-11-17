"""Dataset adapters. Each schema states modalities, labels, and what is unconfirmed."""

from preprocessing.adapters.registry import get_schema, list_datasets

__all__ = ["get_schema", "list_datasets"]
