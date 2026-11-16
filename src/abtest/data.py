"""Download, verify, load and split the Dry Bean dataset."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(force: bool = False) -> None:
    """Download the UCI zip once and verify its checksum."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force or not config.DATASET_ZIP.exists():
        print(f"Downloading {config.DATASET_URL}")
        with urllib.request.urlopen(config.DATASET_URL, timeout=120) as resp:
            config.DATASET_ZIP.write_bytes(resp.read())
    digest = _sha256(config.DATASET_ZIP)
    if digest != config.DATASET_SHA256:
        raise RuntimeError(
            f"Checksum mismatch for {config.DATASET_ZIP}: {digest} != {config.DATASET_SHA256}"
        )


def load_raw() -> pd.DataFrame:
    """Return the raw table exactly as published (13,611 rows)."""
    download()
    with zipfile.ZipFile(config.DATASET_ZIP) as zf:
        payload = zf.read(config.DATASET_MEMBER)
    return pd.read_excel(io.BytesIO(payload))


@dataclass
class Dataset:
    X_dev: pd.DataFrame
    y_dev: np.ndarray
    X_holdout: pd.DataFrame
    y_holdout: np.ndarray
    classes: list[str]
    n_raw: int
    n_duplicates: int

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_dev.columns)


def load_clean() -> tuple[pd.DataFrame, int, int]:
    """Raw table with exact duplicate rows removed."""
    raw = load_raw()
    n_raw = len(raw)
    clean = raw.drop_duplicates().reset_index(drop=True)
    return clean, n_raw, n_raw - len(clean)


def load_dataset() -> Dataset:
    """Clean data, encode labels and make the stratified dev / hold-out split."""
    df, n_raw, n_dup = load_clean()
    classes = sorted(df[config.TARGET].unique())
    y = df[config.TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
    X = df.drop(columns=[config.TARGET]).astype(float)

    X_dev, X_ho, y_dev, y_ho = train_test_split(
        X, y, test_size=config.HOLDOUT_SIZE, stratify=y, random_state=config.SEED
    )
    return Dataset(
        X_dev=X_dev.reset_index(drop=True),
        y_dev=y_dev,
        X_holdout=X_ho.reset_index(drop=True),
        y_holdout=y_ho,
        classes=classes,
        n_raw=n_raw,
        n_duplicates=n_dup,
    )
