"""Read-only access to AIC assets already mounted by Kaggle.

This module intentionally has no dataset-download API. The AIC assets are
very large and must be attached in Kaggle's **Add Input** panel.
"""

from data_paths import AICPaths, DatasetNotFoundError

__all__ = ["AICPaths", "DatasetNotFoundError"]
