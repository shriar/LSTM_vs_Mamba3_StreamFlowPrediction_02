"""CAMELS data download, scaling, and loading utilities.

Two independent paths live here:

* **Legacy loading** (``load_and_scale_data`` and the hydroDL helpers) — kept
  exactly as it was so old code keeps importing.
* **Native acquisition + preprocessing** (B2a, bottom of file) — downloads the
  CAMELS v1.2 record itself, parses the Daymet forcing and USGS streamflow
  text files directly, and writes the ``x / y / c`` arrays the models train
  on, with a checksum manifest.  Build with::

      python src/data_utils.py --build

* **Split + equivalence** (B2b) — writes the 15/4/15 pickles and the tracked
  split spec, then proves the rebuild matches the pre-revision pickles::

      python src/data_utils.py --split
      python src/data_utils.py --verify
      python src/data_utils.py --adjudicate
"""

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
import zipfile
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

from paths import DATA_DIR
from constants import ATTR_LST, forcing_variables

import numpy as np
import pandas as pd


DEFAULT_GDRIVE_FILE_IDS = {
    "training_file": "1z9UBkn86dBIKo1wu1v7bYSTHG87RzGWX",
    "validation_file": "12f-ZrWLxtWRwzacmxBZ23-76_buGhlNa",
}


def download_file(url, destination):
    """Stream a URL to a local file."""
    import requests

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)


def download_google_drive_file(file_id, destination, chunk_size=1024 * 1024):
    """Download a file from Google Drive, handling large file virus-scan prompts.

    Parameters
    ----------
    file_id : str
        Google Drive file ID.
    destination : str or Path
        Target local file path.
    chunk_size : int, optional
        Streaming chunk size in bytes (default 1MB).
    """
    import re
    import requests

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    session = requests.Session()
    url = f"https://drive.google.com/uc?id={file_id}"
    response = session.get(url, stream=True, timeout=60)
    response.raise_for_status()

    # Google Drive redirects large files (>100MB) to an HTML virus-scan confirmation form
    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type:
        text = response.text
        form_action = re.search(r'action="([^"]+)"', text)
        confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', text)
        uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', text)

        if form_action and confirm_match and uuid_match:
            download_url = form_action.group(1)
            params = {
                "id": file_id,
                "confirm": confirm_match.group(1),
                "uuid": uuid_match.group(1),
            }
            response = session.get(download_url, params=params, stream=True, timeout=120)
            response.raise_for_status()
        else:
            # Fallback for alternative warning formats
            token = None
            for key, val in response.cookies.items():
                if key.startswith("download_warning"):
                    token = val
                    break
            if token:
                response = session.get(
                    "https://drive.google.com/uc?export=download",
                    params={"id": file_id, "confirm": token},
                    stream=True,
                    timeout=120,
                )
                response.raise_for_status()
            else:
                m_href = re.search(r'href="(/uc\?[^"]*confirm=[^"]*)"', text)
                if m_href:
                    confirm_url = "https://drive.google.com" + m_href.group(1).replace("&amp;", "&")
                    response = session.get(confirm_url, stream=True, timeout=120)
                    response.raise_for_status()
                else:
                    raise RuntimeError(
                        f"Could not parse Google Drive download confirmation page for file ID '{file_id}'."
                    )

    total_size = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    size_mb = f"{total_size / (1024 * 1024):.1f} MB" if total_size > 0 else "unknown size"
    print(f"Downloading '{destination.name}' from Google Drive ({size_mb})...")

    try:
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        mb = downloaded / (1024 * 1024)
                        tot_mb = total_size / (1024 * 1024)
                        print(f"\r  [{pct:5.1f}%] {mb:.1f} / {tot_mb:.1f} MB", end="", flush=True)
                    else:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  Downloaded {mb:.1f} MB", end="", flush=True)
        if total_size > 0:
            print()
        if temp_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file '{destination.name}' is empty.")
        if destination.exists():
            destination.unlink()
        temp_path.rename(destination)
        print(f"Saved: {destination}")
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def download_gauge_ids(url, destination, n_basins=None):
    """Fetch the compact official CAMELS gauge list for cached-array runs.

    Also downloads camels_topo.txt (lat/lon) from the same Zenodo record
    so that gauge_ids.csv includes gauge_name, gauge_lat, and gauge_lon.
    """
    import requests

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        content = response.content

    metadata = pd.read_csv(StringIO(content.decode("utf-8")), sep=";", dtype={"gauge_id": str})
    if "gauge_id" not in metadata.columns:
        raise ValueError("CAMELS gauge metadata did not contain a gauge_id column.")
    ids = np.asarray([format_gauge_id(value) for value in metadata["gauge_id"]])
    if n_basins is not None and len(ids) != n_basins:
        raise ValueError(
            f"Downloaded CAMELS metadata has {len(ids)} IDs but the data have {n_basins} basins."
        )

    # Build an enriched DataFrame with gauge_name, lat, lon
    extras = {"gauge_name": metadata.get("gauge_name")}

    # Fetch topo file for lat/lon (same Zenodo base as the name file)
    topo_url = url.replace("camels_name.txt", "camels_topo.txt")
    try:
        with requests.get(topo_url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            topo = pd.read_csv(StringIO(resp.content.decode("utf-8")), sep=";",
                               dtype={"gauge_id": str})
        topo["gauge_id"] = topo["gauge_id"].apply(format_gauge_id)
        topo = topo.set_index("gauge_id")
        extras["gauge_lat"] = [topo.loc[gid, "gauge_lat"] if gid in topo.index else np.nan for gid in ids]
        extras["gauge_lon"] = [topo.loc[gid, "gauge_lon"] if gid in topo.index else np.nan for gid in ids]
    except Exception:
        pass  # lat/lon are optional; proceed without them

    return save_gauge_ids(destination, ids, extras=extras)

def download_camels():
    """Download the CAMELS inputs using the layout expected by hydroDL."""
    base_path = DATA_DIR
    archive_base = "https://zenodo.org/records/15529996/files/"
    attributes = base_path / "camels_attributes_v2.0"
    attributes.mkdir(parents=True, exist_ok=True)

    dataset_archive = base_path / "basin_timeseries_v1p2_metForcing_obsFlow.zip"
    download_file(archive_base + "/basin_timeseries_v1p2_metForcing_obsFlow.zip", dataset_archive)
    with zipfile.ZipFile(dataset_archive) as archive:
        archive.extractall(base_path / "basin_timeseries_v1p2_metForcing_obsFlow")

    for name in (
        "camels_attributes_v2.0.xlsx", "camels_clim.txt", "camels_geol.txt",
        "camels_hydro.txt", "camels_name.txt", "camels_soil.txt",
        "camels_topo.txt", "camels_vege.txt",
    ):
        download_file(archive_base + "/" + name, attributes / name)

    pet_archive = base_path / "pet_harg.zip"
    download_file("https://zenodo.org/record/7943626/files/pet_harg.zip?download=1", pet_archive)
    with zipfile.ZipFile(pet_archive) as archive:
        archive.extractall(base_path)
    print("Download and extraction complete.")


def format_gauge_id(value):
    """Zero-pad USGS IDs so 1013500 becomes 01013500."""
    text = str(value).strip()
    if text.isdigit():
        return text.zfill(8)
    return text


def gauge_ids_from_loader(loader, camels_module=None):
    """Pull basin IDs from a hydroDL DataframeCamels instance."""
    for attr in ("usgsId", "idLst", "gageid", "gageId"):
        ids = getattr(loader, attr, None)
        if ids is not None:
            return np.asarray([format_gauge_id(x) for x in ids])
    if camels_module is not None:
        ids = getattr(camels_module, "gageid", None)
        if ids is not None:
            return np.asarray([format_gauge_id(x) for x in ids])
    raise AttributeError(
        "Could not find gauge IDs on the CAMELS loader "
        "(tried usgsId, idLst, gageid, gageId)."
    )


def save_gauge_ids(path, gauge_ids, extras=None):
    """Write USGS gauge IDs (with optional name/lat/lon) to a CSV keyed by basin row index."""
    path = Path(path)
    ids = [format_gauge_id(x) for x in np.asarray(gauge_ids).reshape(-1)]
    df = pd.DataFrame({"gauge_id": ids})
    if extras:
        for col, values in extras.items():
            if values is not None and len(values) == len(ids):
                df[col] = list(values)
    df.to_csv(path, index_label="basin_index")
    print(f"Saved {len(ids)} gauge IDs to {path}")
    return np.asarray(ids)


def load_gauge_ids(path, n_basins=None):
    """Load USGS gauge IDs previously saved by save_gauge_ids."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Gauge ID file not found: {path}. "
            "Re-extract CAMELS (data_opt=1) or create this CSV first."
        )
    df = pd.read_csv(path, dtype={"gauge_id": str})
    if "gauge_id" not in df.columns:
        raise ValueError(f"{path} must contain a gauge_id column.")
    ids = np.asarray([format_gauge_id(x) for x in df["gauge_id"]])
    if n_basins is not None and len(ids) != n_basins:
        raise ValueError(
            f"{path} has {len(ids)} IDs but the data have {n_basins} basins."
        )
    return ids


def extract_camels(t_range, attributes, forcing_variables, camels, *, for_type="daymet",
                   flow_regime=1, subset="All", file_path=None, gauge_id_path=None):
    """Load CAMELS arrays and optionally cache them in a pickle file."""
    loader = camels.DataframeCamels(subset=subset, tRange=t_range, forType=for_type)
    data = (
        loader.getDataTs(varLst=forcing_variables, doNorm=False, rmNan=False, flow_regime=flow_regime),
        loader.getDataObs(doNorm=False, rmNan=False, basinnorm=False, flow_regime=flow_regime),
        loader.getDataConst(varLst=attributes, doNorm=False, rmNan=False, flow_regime=flow_regime),
    )
    if file_path is not None:
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
    gauge_ids = gauge_ids_from_loader(loader, camels)
    if gauge_id_path is not None:
        save_gauge_ids(gauge_id_path, gauge_ids)
    return data, gauge_ids


def _statistics(values, log_transform=False):
    values = values.flatten()
    values = values[~np.isnan(values)]
    if log_transform:
        values = np.log10(np.sqrt(values) + 0.1)
    std = np.std(values).astype(float)
    return [
        np.percentile(values, 10).astype(float),
        np.percentile(values, 90).astype(float),
        np.mean(values).astype(float),
        std if std >= 0.001 else 1,
    ]


def get_statistics(log_norm_columns, *, attribute_names=None, attribute_data=None,
                   series_names=None, series_data=None):
    """Calculate normalization statistics for time series and static attributes."""
    statistics = {}
    if series_names is not None:
        for index, name in enumerate(series_names):
            statistics[name] = _statistics(series_data[:, :, index], name in log_norm_columns)
    if attribute_names is not None:
        for index, name in enumerate(attribute_names):
            statistics[name] = _statistics(attribute_data[:, index])
    return statistics


def transform_by_statistics(data, variables, statistics, log_norm_columns, normalize):
    """Normalize or invert-normalize a 2-D or 3-D array."""
    variables = [variables] if isinstance(variables, str) else variables
    values, output = data.copy(), np.full(data.shape, np.nan)
    is_series = data.ndim == 3
    for index, name in enumerate(variables):
        mean, std = statistics[name][2:]
        target = (slice(None), slice(None), index) if is_series else (slice(None), index)
        if normalize:
            if name in log_norm_columns:
                values[target] = np.log10(np.sqrt(values[target]) + 0.1)
            output[target] = (values[target] - mean) / std
        else:
            output[target] = values[target] * std + mean
            if name in log_norm_columns:
                output[target] = (np.power(10, output[target]) - 0.1) ** 2
    return output


class HydroScaler:
    """CAMELS normalization wrapper with a scikit-learn-like interface."""

    def __init__(self, attrLst, seriesLst, xNanFill, log_norm_cols):
        self.attrLst = attrLst
        self.seriesLst = seriesLst
        self.xNanFill = xNanFill
        self.log_norm_cols = log_norm_cols
        self.stat_dict = None

    def fit(self, attribute_data, series_data):
        self.stat_dict = get_statistics(
            self.log_norm_cols, attribute_names=self.attrLst, attribute_data=attribute_data,
            series_names=self.seriesLst, series_data=series_data,
        )

    def transform(self, data, variables):
        return transform_by_statistics(data, variables, self.stat_dict, self.log_norm_cols, True)

    def fit_transform(self, attribute_data, series_data):
        self.fit(attribute_data, series_data)
        return self.transform(attribute_data, self.attrLst), self.transform(series_data, self.seriesLst)


def basin_norm(values, basin_area, mean_precipitation, normalize):
    """Convert streamflow between ft³/s and dimensionless basin-normalized flow."""
    dimensions = values.ndim
    if dimensions == 3 and values.shape[2] == 1:
        values = values[:, :, 0]
    area = np.tile(basin_area, (1, values.shape[1]))
    precipitation = np.tile(mean_precipitation, (1, values.shape[1]))
    conversion = (area * 10**6) * (precipitation * 10**-3)
    flow = values * 0.0283168 * 3600 * 24 / conversion if normalize else values * conversion / (0.0283168 * 3600 * 24)
    return np.expand_dims(flow, axis=2) if dimensions == 3 else flow


# Keep the tutorial's original public names while centralizing their implementation.
getStatDic = get_statistics
calcStat = _statistics


def calcStatgamma(values):
    return _statistics(values, log_transform=True)


def transNormbyDic(x_in, var_lst, stat_dict, log_norm_cols, to_norm):
    return transform_by_statistics(x_in, var_lst, stat_dict, log_norm_cols, to_norm)


def basinNorm(x, basinarea, meanprep, toNorm):
    return basin_norm(x, basinarea, meanprep, toNorm)


def ensure_gauge_ids(cfg, n_basins=None, loader=None, camels_module=None):
    """Load gauge IDs from CSV, or create the CSV from hydroDL on first use."""
    path = cfg.gauge_id_path()
    if path.exists():
        return load_gauge_ids(path, n_basins=n_basins)
    if loader is not None:
        ids = gauge_ids_from_loader(loader, camels_module)
        return save_gauge_ids(path, ids)

    # Preprocessed arrays do not contain their basin IDs.  Fetch the small
    # official name table instead of requiring the full CAMELS time-series
    # archive just to recover this mapping.
    try:
        return download_gauge_ids(cfg.gauge_metadata_url, path, n_basins=n_basins)
    except Exception as metadata_error:
        raise FileNotFoundError(
            f"Gauge ID file not found: {path}. Downloading the official CAMELS "
            f"gauge metadata also failed: {metadata_error}"
        ) from metadata_error



def load_and_scale_data(cfg):
    """Load CAMELS data, fit scaler, and return everything the main scripts need.

    Parameters
    ----------
    cfg : config.ExperimentConfig

    Returns
    -------
    dict with keys:
        train_x, train_y, train_c, val_x, val_y, val_c,
        scaler, basinarea, meanprep, gauge_ids
    """
    from hydroDL.master import default
    from hydroDL.data import camels

    opt_data = default.optDataCamels
    opt_data = default.update(
        opt_data, varT=cfg.varF, varC=cfg.attrLst,
        tRange=cfg.t_train, forType=cfg.for_type, subset=cfg.subset_train,
    )

    # Resolve preprocessed file paths (prefer existing file, otherwise script_dir)
    train_path = getattr(cfg, "train_path", lambda: None)()
    if train_path is None:
        p = Path(cfg.train_file)
        train_path = p if (p.is_absolute() or p.exists()) else cfg.script_dir / cfg.train_file

    val_path = getattr(cfg, "validation_path", lambda: None)()
    if val_path is None:
        p = Path(cfg.validation_file)
        val_path = p if (p.is_absolute() or p.exists()) else cfg.script_dir / cfg.validation_file

    if cfg.data_opt == 1:
        download_camels()
        just_load = False
    elif cfg.data_opt == 2:
        just_load = True
        file_specs = [
            (train_path, getattr(cfg, "train_file_gdrive_id", DEFAULT_GDRIVE_FILE_IDS.get("training_file"))),
            (val_path, getattr(cfg, "validation_file_gdrive_id", DEFAULT_GDRIVE_FILE_IDS.get("validation_file"))),
        ]
        for path, gdrive_id in file_specs:
            if not path.exists():
                if gdrive_id:
                    print(f"Preprocessed CAMELS dataset '{path.name}' not found locally.")
                    print(f"Auto-downloading from Google Drive (ID: {gdrive_id})...")
                    download_google_drive_file(gdrive_id, path)
                else:
                    raise FileNotFoundError(
                        f"Missing preprocessed CAMELS file: {path}. No Google Drive ID specified."
                    )
    else:
        just_load = True

    gauge_id_path = cfg.gauge_id_path()

    if just_load:
        with open(train_path, "rb") as f:
            train_x, train_y, train_c = pickle.load(f)
        with open(val_path, "rb") as f:
            val_x, val_y, val_c = pickle.load(f)
        gauge_ids = ensure_gauge_ids(cfg, n_basins=val_y.shape[0])
    else:
        camels.initcamels(flow_regime=cfg.flow_regime, forType=cfg.for_type,
                          rootDB=str(DATA_DIR))
        (train_x, train_y, train_c), gauge_ids = extract_camels(
            cfg.t_train, cfg.attrLst, cfg.varF, camels,
            for_type=cfg.for_type, flow_regime=cfg.flow_regime,
            subset=cfg.subset_train, file_path=train_path,
            gauge_id_path=gauge_id_path,
        )
        (val_x, val_y, val_c), _ = extract_camels(
            cfg.t_valid, cfg.attrLst, cfg.varF, camels,
            for_type=cfg.for_type, flow_regime=cfg.flow_regime,
            subset=cfg.subset_train, file_path=val_path,
            gauge_id_path=gauge_id_path,
        )

    # Basin normalization for dimensionless streamflow
    basinarea = val_c[:, np.where(np.array(cfg.attrLst) == "area_gages2")[0]]
    meanprep = val_c[:, np.where(np.array(cfg.attrLst) == "p_mean")[0]]
    y_temp = basin_norm(train_y, basinarea, meanprep, True)

    series_data = np.concatenate([train_x, y_temp], axis=2)
    series_vars = cfg.varF + ["runoff"]

    # Fit scaler
    scaler = HydroScaler(
        attrLst=cfg.attrLst, seriesLst=series_vars,
        xNanFill=0.0, log_norm_cols=cfg.log_norm_cols,
    )
    attr_norm, series_norm = scaler.fit_transform(train_c, series_data)
    attr_norm[np.isnan(attr_norm)] = 0.0

    train_x = series_norm[:, :, :-1]
    train_x[np.isnan(train_x)] = 0.0
    train_y = np.expand_dims(series_norm[:, :, -1], 2)
    train_c = attr_norm

    val_c = scaler.transform(val_c, cfg.attrLst)
    val_x = scaler.transform(val_x, cfg.varF)
    val_c[np.isnan(val_c)] = 0.0
    val_x[np.isnan(val_x)] = 0.0

    print(f"Data loaded: train_x={train_x.shape} train_y={train_y.shape} train_c={train_c.shape}")

    return {
        "train_x": train_x, "train_y": train_y, "train_c": train_c,
        "val_x": val_x, "val_y": val_y, "val_c": val_c,
        "scaler": scaler, "basinarea": basinarea, "meanprep": meanprep,
        "gauge_ids": gauge_ids, "opt_data": opt_data,
    }


# ===========================================================================
# B2a — native CAMELS acquisition and preprocessing
# ===========================================================================
#
# Everything below is deliberately independent of hydroDL: it downloads the
# published CAMELS record, verifies it, parses the plain-text forcing and
# streamflow files, and writes the arrays the models consume.  Nothing here
# touches the train/val/test split (that is B2b); the scaler below is fitted
# on the training window only, and the window constants live next to the
# split so there is exactly one definition of each date.

CAMELS_ZENODO_RECORD = 15529996
CAMELS_ZENODO_BASE = f"https://zenodo.org/records/{CAMELS_ZENODO_RECORD}/files/"
CAMELS_VERSION = "1.2"
CAMELS_DATASET_TITLE = (
    "CAMELS: Catchment Attributes and MEteorology for Large-sample Studies"
)

TIMESERIES_ZIP = "basin_timeseries_v1p2_metForcing_obsFlow.zip"

# The seven attribute tables.  `camels_name.txt` carries the gauge metadata
# (gauge_ids.csv was derived from it); the other six are read by
# `read_attributes()`.
ATTRIBUTE_TABLES = (
    "camels_clim.txt",
    "camels_geol.txt",
    "camels_hydro.txt",
    "camels_name.txt",
    "camels_soil.txt",
    "camels_topo.txt",
    "camels_vege.txt",
)

# Published size + MD5, read from the Zenodo record API on 2026-10-28
# (https://zenodo.org/api/records/15529996).  Zenodo publishes MD5 only, so
# the download is verified against these; SHA256 is computed locally and kept
# in `data/manifest.json` for provenance.
CAMELS_FILE_SPECS = {
    TIMESERIES_ZIP: {"bytes": 3406626583, "md5": "8e9a466710e8270b58f01d332a87184f"},
    "camels_clim.txt": {"bytes": 100673, "md5": "67f22592f3fb72c57df81358ce68458b"},
    "camels_geol.txt": {"bytes": 71583, "md5": "f5ce5de53eb1ea2532cda7e3b4813993"},
    "camels_hydro.txt": {"bytes": 122799, "md5": "55ebdeb36c42ee7acdb998229c3edb3a"},
    "camels_name.txt": {"bytes": 30417, "md5": "c96491b32c4df55a31bead7ceca7d64b"},
    "camels_soil.txt": {"bytes": 109125, "md5": "8edb46a363a20b466a4b7105ba633767"},
    "camels_topo.txt": {"bytes": 38677, "md5": "0f6267838c40b1507b64582433bc0b8e"},
    "camels_vege.txt": {"bytes": 107970, "md5": "f40e843defc1e654a800be9fe5fd5090"},
}

# 34 water years, 1980-10-01 → 2014-09-30 inclusive.
DATASET_START = "1980-10-01"
DATASET_END = "2014-09-30"
DATASET_DAYS = 12418
EXPECTED_BASINS = 671

# Forcing channels, in the model's input order.  `tmean` is not a Daymet
# column: it is (tmax + tmin) / 2, the definition hydroDL used when it built
# the legacy pickles (`hydroDL/data/camels.py`, `getDataTs`).
FORCING_ORDER = tuple(forcing_variables("daymet"))
FORCING_UNITS = {
    "dayl": "s",
    "prcp": "mm/day",
    "srad": "W/m2",
    "tmean": "degC",
    "vp": "Pa",
}

# One definition of the split, consumed by B2a (scaler fit) and B2b (pickles).
SPLIT_WINDOWS = {
    "train": ("1980-10-01", "1995-09-30"),
    "val": ("1995-10-01", "1999-09-30"),
    "test": ("1999-10-01", "2014-09-30"),
}
CONTEXT_DAYS = 730  # rho 365 + train_buff 365, borrowed from the previous slice

RAW_DIR = DATA_DIR / "raw"
ATTRIBUTE_DIR = DATA_DIR / "camels_attributes_v2.0"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
MANIFEST_PATH = DATA_DIR / "manifest.json"
LAYOUT_PATH = RAW_DIR / "layout.json"
ZIP_PATH = RAW_DIR / TIMESERIES_ZIP
EXTRACT_DIR = RAW_DIR / "basin_timeseries_v1p2_metForcing_obsFlow"
EXTRACT_MARKER = RAW_DIR / ".extract.json"
GAUGE_IDS_PATH = DATA_DIR / "gauge_ids.csv"

FORCING_ROOT = Path("basin_dataset_public_v1p2") / "basin_mean_forcing" / "daymet"
FLOW_ROOT = Path("basin_dataset_public_v1p2") / "usgs_streamflow"
METADATA_ROOT = Path("basin_dataset_public_v1p2") / "basin_metadata"

# Sentinel used by the CAMELS forcing files for missing values.
FORCING_SENTINEL = -999.0

# CAMELS v2.0 ships `geol_porostiy` (typo preserved upstream); the model's
# attribute list uses the corrected spelling.  Map one to the other here so
# the spelling stays fixed in exactly one place.
ATTRIBUTE_ALIASES = {"geol_porosity": "geol_porostiy"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} TB"


def _rel(path):
    """Path relative to the experiment root, for stable manifest entries."""
    try:
        return Path(path).resolve().relative_to(DATA_DIR.parent).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _md5_file(path, chunk_size=1024 * 1024):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_if_changed(path, payload):
    """Write canonical JSON, but leave the file untouched when identical.

    This is what makes a second `--build` a genuine no-op: no rewritten file,
    no churned timestamp, no dirty git tree.
    """
    path = Path(path)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == text:
                return False
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return True


def _read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _gauge_ids_sha256(gauge_ids):
    return hashlib.sha256("\n".join(str(g) for g in gauge_ids).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Step 1 — download (resumable, checksum-verified)
# ---------------------------------------------------------------------------

def _download_resumable(url, destination, spec, *, force=False, retries=4,
                        previous=None):
    """Download one file, resuming a partial transfer, then verify size + MD5.

    Returns a manifest entry.  A file that already matches its published MD5
    is left untouched (and its original retrieval date preserved).
    """
    import requests

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")

    if destination.exists() and not force:
        size = destination.stat().st_size
        known = (previous or {}).get("md5")
        if size == spec["bytes"] and known == spec["md5"]:
            # Trust the manifest's recorded checksum: re-hashing 3.4 GB on
            # every cached run is exactly what a no-op must not do.
            print(f"  cached   {destination.name} ({_fmt_bytes(size)}, md5 ok)")
            return {
                "key": destination.name,
                "url": url,
                "local_path": _rel(destination),
                "bytes": size,
                "bytes_expected": spec["bytes"],
                "md5": spec["md5"],
                "sha256": (previous or {}).get("sha256") or _sha256_file(destination),
                "retrieved_utc": (previous or {}).get("retrieved_utc") or _utc_now(),
                "state": "cached",
            }
        if size != spec["bytes"]:
            print(
                f"  partial  {destination.name}: {_fmt_bytes(size)} on disk, "
                f"{_fmt_bytes(spec['bytes'])} published — re-downloading"
            )
        else:
            digest = _md5_file(destination)
            if digest != spec["md5"]:
                print(f"  corrupt  {destination.name}: md5 {digest} != {spec['md5']}")
        if destination.exists():
            destination.unlink()

    if destination.exists():
        size = destination.stat().st_size
        return {
            "key": destination.name,
            "url": url,
            "local_path": _rel(destination),
            "bytes": size,
            "bytes_expected": spec["bytes"],
            "md5": spec["md5"],
            "sha256": _sha256_file(destination),
            "retrieved_utc": (previous or {}).get("retrieved_utc") or _utc_now(),
            "state": "cached",
        }

    attempt = 0
    while attempt < retries:
        attempt += 1
        have = part.stat().st_size if part.exists() else 0
        if have > spec["bytes"]:
            part.unlink()
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        print(
            f"  fetch    {destination.name} "
            f"({_fmt_bytes(spec['bytes'])})"
            + (f" — resuming at {_fmt_bytes(have)}" if have else "")
            + (f" [attempt {attempt}/{retries}]" if attempt > 1 else "")
        )
        try:
            with requests.get(url, stream=True, timeout=(30, 180), headers=headers) as response:
                if response.status_code == 416 and have:
                    # Already complete at the byte level; fall through to verify.
                    response.close()
                else:
                    response.raise_for_status()
                    if have and response.status_code != 206:
                        # Server ignored Range: restart from zero.
                        have = 0
                        part.unlink(missing_ok=True)
                    mode = "ab" if have else "wb"
                    total = spec["bytes"]
                    done = have
                    started = time.time()
                    last_print = 0.0
                    with open(part, mode) as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            done += len(chunk)
                            now = time.time()
                            if now - last_print >= 2.0:
                                last_print = now
                                elapsed = max(now - started, 1e-6)
                                rate = (done - have) / elapsed / (1024 * 1024)
                                pct = done / total * 100 if total else 0.0
                                remaining = (total - done) / max(rate * 1024 * 1024, 1.0)
                                print(
                                    f"\r    {done / 1e9:6.2f} / {total / 1e9:6.2f} GB"
                                    f"  {rate:6.1f} MB/s  {pct:5.1f}%"
                                    f"  eta {remaining / 60:5.1f} min ",
                                    end="",
                                    flush=True,
                                )
                    print()
        except Exception as error:  # noqa: BLE001 - retried below
            print(f"    download error: {error}")
            if attempt >= retries:
                raise
            time.sleep(min(5 * attempt, 20))
            continue

        size = part.stat().st_size if part.exists() else 0
        if size != spec["bytes"]:
            print(f"    size mismatch: {size} != {spec['bytes']}")
            if attempt >= retries:
                raise RuntimeError(f"{destination.name}: incomplete download")
            continue
        digest = _md5_file(part)
        if digest != spec["md5"]:
            print(f"    md5 mismatch: {digest} != {spec['md5']} — discarding partial file")
            part.unlink(missing_ok=True)
            if attempt >= retries:
                raise RuntimeError(f"{destination.name}: checksum mismatch")
            continue
        os.replace(part, destination)
        print(f"  saved    {destination.name} ({_fmt_bytes(size)}, md5 ok)")
        return {
            "key": destination.name,
            "url": url,
            "local_path": _rel(destination),
            "bytes": size,
            "bytes_expected": spec["bytes"],
            "md5": digest,
            "sha256": _sha256_file(destination),
            "retrieved_utc": _utc_now(),
            "state": "downloaded",
        }

    raise RuntimeError(f"{destination.name}: download failed after {retries} attempts")


def download_raw(force=False):
    """Download the time-series archive and the seven attribute tables.

    Skips the three VIC ``modelOutput`` archives (11 GB) and
    ``basin_set_full_res.zip``; those were never model inputs.  Returns the
    list of manifest entries.
    """
    print("[1/5] download CAMELS v1.2 record")
    previous = {entry["key"]: entry
                for entry in (_read_json(MANIFEST_PATH, {}) or {}).get("files", [])}
    entries = []
    for key in (TIMESERIES_ZIP,) + ATTRIBUTE_TABLES:
        spec = CAMELS_FILE_SPECS[key]
        destination = ZIP_PATH if key == TIMESERIES_ZIP else ATTRIBUTE_DIR / key
        entries.append(
            _download_resumable(
                CAMELS_ZENODO_BASE + key,
                destination,
                spec,
                force=force,
                previous=previous.get(key),
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Step 2 — probe the archive before extracting
# ---------------------------------------------------------------------------

def _gauge_from_member(member):
    """`.../01013500_lump_cida_forcing_leap.txt` -> `01013500`."""
    return Path(member).name.split("_", 1)[0]


def _huc_from_member(member):
    """`.../daymet/01/01013500_...` -> `01`."""
    return Path(member).parts[-2]


def probe_layout(force=False):
    """List the archive and record what is actually inside it.

    Never assume the HUC folder names, the file suffix, or that every basin
    has both files: this writes the observed structure to
    ``data/raw/layout.json`` and fails loudly on gaps.
    """
    print("[2/5] probe archive layout")
    gauge_ids = load_gauge_ids(GAUGE_IDS_PATH, n_basins=EXPECTED_BASINS)
    if not force:
        cached = _read_json(LAYOUT_PATH)
        if cached and cached.get("zip", {}).get("md5") == CAMELS_FILE_SPECS[TIMESERIES_ZIP]["md5"]:
            coverage = cached.get("coverage", {})
            if coverage.get("missing_forcing") == [] and coverage.get("missing_streamflow") == []:
                print(f"  cached   {_rel(LAYOUT_PATH)} "
                      f"({coverage.get('forcing')} forcing + "
                      f"{coverage.get('streamflow')} streamflow members)")
                return cached

    with zipfile.ZipFile(ZIP_PATH) as archive:
        members = archive.namelist()

    forcing = [m for m in members if m.endswith("_lump_cida_forcing_leap.txt")]
    streamflow = [m for m in members if m.endswith("_streamflow_qc.txt")]
    metadata = [m for m in members if "/basin_metadata/" in m]

    forcing_ids = {_gauge_from_member(m) for m in forcing}
    streamflow_ids = {_gauge_from_member(m) for m in streamflow}
    expected = set(gauge_ids)
    missing_forcing = sorted(expected - forcing_ids)
    missing_streamflow = sorted(expected - streamflow_ids)
    extra_forcing = sorted(forcing_ids - expected)

    top_level = sorted({m.split("/", 1)[0] for m in members if m})
    forcing_hucs = sorted({_huc_from_member(m) for m in forcing})
    flow_hucs = sorted({_huc_from_member(m) for m in streamflow})

    layout = {
        "zip": {
            "name": TIMESERIES_ZIP,
            "bytes": CAMELS_FILE_SPECS[TIMESERIES_ZIP]["bytes"],
            "md5": CAMELS_FILE_SPECS[TIMESERIES_ZIP]["md5"],
        },
        "n_members": len(members),
        "top_level": top_level,
        "sections": {
            "forcing": {
                "count": len(forcing),
                "template": (FORCING_ROOT / "{huc:02d}"
                             / "{gauge_id}_lump_cida_forcing_leap.txt").as_posix(),
                "hucs": forcing_hucs,
            },
            "streamflow": {
                "count": len(streamflow),
                "template": (FLOW_ROOT / "{huc:02d}"
                             / "{gauge_id}_streamflow_qc.txt").as_posix(),
                "hucs": flow_hucs,
            },
            "metadata": {
                "count": len(metadata),
                "members": metadata[:20],
            },
        },
        "coverage": {
            "expected": len(expected),
            "forcing": len(forcing_ids),
            "streamflow": len(streamflow_ids),
            "missing_forcing": missing_forcing,
            "missing_streamflow": missing_streamflow,
            "extra_forcing": extra_forcing,
        },
        "sample_members": members[:5] + members[-5:],
    }
    _write_json_if_changed(LAYOUT_PATH, layout)
    print(f"  found    {len(forcing)} forcing, {len(streamflow)} streamflow, "
          f"{len(members)} members across {len(top_level)} top-level dirs")
    if missing_forcing or missing_streamflow:
        raise RuntimeError(
            "Archive is missing basins from data/gauge_ids.csv: "
            f"{len(missing_forcing)} forcing, {len(missing_streamflow)} streamflow. "
            f"See {LAYOUT_PATH}."
        )
    return layout


# ---------------------------------------------------------------------------
# Step 3 — extract (once, tracked by a marker)
# ---------------------------------------------------------------------------

def extract_raw(force=False):
    """Extract the archive under ``data/raw/``, skipping a completed extraction."""
    print("[3/5] extract archive")
    marker = _read_json(EXTRACT_MARKER, {}) or {}
    zip_md5 = CAMELS_FILE_SPECS[TIMESERIES_ZIP]["md5"]
    probe = EXTRACT_DIR / FORCING_ROOT
    if (not force and marker.get("zip_md5") == zip_md5
            and marker.get("n_members") and probe.is_dir()):
        print(f"  cached   {_rel(EXTRACT_DIR)} "
              f"({marker['n_members']} members, {_fmt_bytes(marker.get('bytes', 0))})")
        return marker

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total_bytes = 0
    with zipfile.ZipFile(ZIP_PATH) as archive:
        infos = archive.infolist()
        for index, info in enumerate(infos, 1):
            archive.extract(info, EXTRACT_DIR)
            total_bytes += info.file_size
            if index % 100 == 0 or index == len(infos):
                elapsed = max(time.time() - started, 1e-6)
                print(f"\r    {index:4d} / {len(infos)} members"
                      f"  {total_bytes / 1e9:5.2f} GB"
                      f"  {total_bytes / elapsed / 1e6:5.1f} MB/s", end="", flush=True)
    print()
    marker = {
        "zip_md5": zip_md5,
        "n_members": len(infos),
        "bytes": total_bytes,
        "extracted_utc": _utc_now(),
    }
    _write_json_if_changed(EXTRACT_MARKER, marker)
    print(f"  done     {_rel(EXTRACT_DIR)} ({len(infos)} members, "
          f"{_fmt_bytes(total_bytes)})")
    return marker


# ---------------------------------------------------------------------------
# Step 4 — parse + build arrays
# ---------------------------------------------------------------------------

def read_basin_table():
    """Official CAMELS gauge metadata: one row per basin with its HUC folder."""
    path = EXTRACT_DIR / METADATA_ROOT / "gauge_information.txt"
    table = pd.read_csv(path, sep="\t", dtype=str)
    table.columns = [column.strip().lower() for column in table.columns]
    if "huc" not in table.columns or "gauge_id" not in table.columns:
        # Some mirrors ship without a usable header; match hydroDL's reader.
        table = pd.read_csv(path, sep="\t", header=None, skiprows=1, dtype=str)
        table = table.iloc[:, :6]
        table.columns = ["huc", "gauge_id", "gauge_name", "lat", "lon", "area"]
    table["gauge_id"] = table["gauge_id"].map(format_gauge_id)
    table["huc"] = table["huc"].str.strip().str.zfill(2)
    return table[["gauge_id", "huc", "gauge_name", "lat", "lon", "area"]]


def _date_axis():
    dates = pd.date_range(DATASET_START, DATASET_END, freq="D")
    if len(dates) != DATASET_DAYS:
        raise RuntimeError(
            f"{DATASET_START} → {DATASET_END} has {len(dates)} days, "
            f"expected {DATASET_DAYS}."
        )
    return dates


def read_forcing_file(path, dates):
    """Daymet forcing for one basin as (len(dates), 5) float64.

    Columns in the file: Year, Mnth, Day, Hr, dayl, prcp, srad, swe, tmax,
    tmin, vp.  The model takes ``dayl, prcp, srad, tmean=(tmax+tmin)/2, vp``.
    Missing rows become NaN; the -999 sentinel is converted to NaN and counted
    by the caller.
    """
    raw = pd.read_csv(path, sep=r"\s+", header=None, skiprows=4)
    if raw.shape[1] < 11:
        raise ValueError(f"{path}: expected 11 columns, found {raw.shape[1]}")
    index = pd.to_datetime({"year": raw[0], "month": raw[1], "day": raw[2]})
    frame = pd.DataFrame(
        {
            "dayl": raw[4].to_numpy(dtype=np.float64),
            "prcp": raw[5].to_numpy(dtype=np.float64),
            "srad": raw[6].to_numpy(dtype=np.float64),
            "tmax": raw[8].to_numpy(dtype=np.float64),
            "tmin": raw[9].to_numpy(dtype=np.float64),
            "vp": raw[10].to_numpy(dtype=np.float64),
        },
        index=index,
    )
    frame = frame[~frame.index.duplicated(keep="first")].reindex(dates)
    out = np.empty((len(dates), len(FORCING_ORDER)), dtype=np.float64)
    out[:, 0] = frame["dayl"].to_numpy()
    out[:, 1] = frame["prcp"].to_numpy()
    out[:, 2] = frame["srad"].to_numpy()
    out[:, 3] = 0.5 * (frame["tmax"].to_numpy() + frame["tmin"].to_numpy())
    out[:, 4] = frame["vp"].to_numpy()
    return out


def read_streamflow_file(path, dates):
    """USGS observed streamflow for one basin, in cfs, as (len(dates),) float64.

    Columns: gauge, Year, Mnth, Day, Qobs, flag.  Negative values are the
    CAMELS missing-data flag and become NaN (same rule as hydroDL).
    """
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    if raw.shape[1] < 5:
        raise ValueError(f"{path}: expected at least 5 columns, found {raw.shape[1]}")
    index = pd.to_datetime({"year": raw[1], "month": raw[2], "day": raw[3]})
    values = raw[4].to_numpy(dtype=np.float64, copy=True)
    values[values < 0] = np.nan
    series = pd.Series(values, index=index)
    series = series[~series.index.duplicated(keep="first")].reindex(dates)
    return series.to_numpy(dtype=np.float64)


def read_attributes(gauge_ids):
    """The 35 static attributes in ``ATTR_LST`` column order.

    String columns (land cover, geology class) are factorized with
    ``pd.factorize(sort=True)`` — exactly how hydroDL encoded them — so the
    result is numeric float64 with -1 for missing.
    """
    table = pd.DataFrame(index=pd.Index(gauge_ids, name="gauge_id"))
    found = {}
    for key in ("clim", "topo", "vege", "soil", "geol"):
        path = ATTRIBUTE_DIR / f"camels_{key}.txt"
        frame = pd.read_csv(path, sep=";", dtype={"gauge_id": str})
        frame["gauge_id"] = frame["gauge_id"].map(format_gauge_id)
        frame = frame.set_index("gauge_id")
        for column in ATTR_LST:
            source = ATTRIBUTE_ALIASES.get(column, column)
            if source in frame.columns:
                table[column] = frame[source].reindex(table.index)
                found[column] = key
    missing = [column for column in ATTR_LST if column not in table.columns]
    if missing:
        raise RuntimeError(
            "Attribute tables are missing columns required by ATTR_LST: "
            f"{missing}. Present: {sorted(found)}"
        )
    table = table[ATTR_LST]
    encoded = {}
    for column in table.columns:
        if pd.api.types.is_string_dtype(table[column]) or table[column].dtype == object:
            codes, uniques = pd.factorize(table[column], sort=True)
            table[column] = codes
            encoded[column] = [str(value) for value in uniques[:5]]
    return table.to_numpy(dtype=np.float64), encoded


def _train_indices(dates):
    start, end = SPLIT_WINDOWS["train"]
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return np.flatnonzero(mask)


def fit_scaler(x, y, c, dates, train_only=True):
    """Fit normalisation on the training window only.

    Mirrors the legacy ``load_and_scale_data`` procedure exactly: streamflow
    is first made dimensionless with ``basin_norm`` (area + p_mean), then the
    series statistics (p10, p90, mean, std) are computed over
    ``[forcing..., runoff]`` and the 35 attributes.

    ``train_only`` exists to make the constraint explicit: fitting on val or
    test days would leak the evaluation period into the loss scale, so any
    other value raises rather than silently doing it.
    """
    if not train_only:
        raise ValueError("fit_scaler refuses to fit outside the training window")
    train_idx = _train_indices(dates)
    if len(train_idx) == 0:
        raise RuntimeError("Training window contains no days.")
    area = c[:, ATTR_LST.index("area_gages2")][:, None]
    p_mean = c[:, ATTR_LST.index("p_mean")][:, None]
    runoff = basin_norm(y[:, train_idx, :], area, p_mean, True)
    series = np.concatenate([x[:, train_idx, :], runoff], axis=2)
    series_names = list(FORCING_ORDER) + ["runoff"]
    stats = get_statistics(
        [],
        attribute_names=ATTR_LST,
        attribute_data=c,
        series_names=series_names,
        series_data=series,
    )
    return {
        "fitted_on": {
            "slice": "train",
            "start": SPLIT_WINDOWS["train"][0],
            "end": SPLIT_WINDOWS["train"][1],
            "days": int(len(train_idx)),
            "basins": int(x.shape[0]),
            "fields": "forcing, basin-normalised runoff, 35 attributes",
        },
        "flow_regime": 1,
        "log_norm_columns": [],
        "series_order": series_names,
        "series_statistics": {name: stats[name] for name in series_names},
        "attribute_statistics": {name: stats[name] for name in ATTR_LST},
        "basin_norm": {
            "area_units": "km2",
            "p_mean_units": "mm/day",
            "target": "runoff (dimensionless)",
            "formula": "y_cfs * 0.0283168 * 86400 / (area_km2 * 1e6 * p_mean_mm * 1e-3)",
            "area_gages2": area.reshape(-1).tolist(),
            "p_mean": p_mean.reshape(-1).tolist(),
        },
        "statistic_definition": "[p10, p90, mean, std] over non-NaN values, std floored at 0.001",
    }


def build_arrays(force=False):
    """Parse every basin and write ``data/preprocessed/{x,y,c}.npy``.

    Returns a summary dict (shapes, NaNs, checks).  A cached build is reused
    when its source archive and basin order match.
    """
    print("[4/5] build arrays (671 basins × 12 418 days)")
    gauge_ids = load_gauge_ids(GAUGE_IDS_PATH, n_basins=EXPECTED_BASINS)
    gauge_sha = _gauge_ids_sha256(gauge_ids)
    zip_md5 = CAMELS_FILE_SPECS[TIMESERIES_ZIP]["md5"]
    meta_path = PREPROCESSED_DIR / "meta.json"
    cached = _read_json(meta_path)
    array_paths = {name: PREPROCESSED_DIR / f"{name}.npy" for name in ("x", "y", "c")}

    if (not force and cached
            and cached.get("source", {}).get("zip_md5") == zip_md5
            and cached.get("basins", {}).get("gauge_ids_sha256") == gauge_sha
            and all(path.exists() for path in array_paths.values())):
        shapes = cached.get("arrays", {})
        print(f"  cached   x={tuple(shapes['x']['shape'])} "
              f"y={tuple(shapes['y']['shape'])} c={tuple(shapes['c']['shape'])}")
        return cached

    dates = _date_axis()
    basins = read_basin_table()
    by_id = basins.set_index("gauge_id")
    missing = [gid for gid in gauge_ids if gid not in by_id.index]
    if missing:
        raise RuntimeError(f"{len(missing)} gauge IDs missing from gauge_information.txt: {missing[:5]}")

    n_basins = len(gauge_ids)
    x = np.empty((n_basins, len(dates), len(FORCING_ORDER)), dtype=np.float64)
    y = np.empty((n_basins, len(dates), 1), dtype=np.float64)
    sentinel_hits = 0
    started = time.time()

    for row, gauge_id in enumerate(gauge_ids):
        huc = by_id.at[gauge_id, "huc"]
        forcing_path = EXTRACT_DIR / FORCING_ROOT / huc / f"{gauge_id}_lump_cida_forcing_leap.txt"
        flow_path = EXTRACT_DIR / FLOW_ROOT / huc / f"{gauge_id}_streamflow_qc.txt"
        if not forcing_path.exists() or not flow_path.exists():
            raise FileNotFoundError(
                f"basin {gauge_id} (HUC {huc}) is missing "
                f"{forcing_path if not forcing_path.exists() else flow_path}"
            )
        forcing = read_forcing_file(forcing_path, dates)
        hits = int(np.count_nonzero(forcing == FORCING_SENTINEL))
        if hits:
            sentinel_hits += hits
            forcing[forcing == FORCING_SENTINEL] = np.nan
        x[row] = forcing
        y[row, :, 0] = read_streamflow_file(flow_path, dates)
        if (row + 1) % 50 == 0 or row == n_basins - 1:
            elapsed = max(time.time() - started, 1e-6)
            print(f"\r    {row + 1:3d} / {n_basins} basins"
                  f"  {elapsed:6.1f} s  {(row + 1) / elapsed:5.1f} basin/s",
                  end="", flush=True)
    print()

    c, encoded = read_attributes(gauge_ids)
    if c.shape != (n_basins, len(ATTR_LST)):
        raise RuntimeError(f"attributes have shape {c.shape}, expected {(n_basins, len(ATTR_LST))}")

    # ---- checks that fail the build rather than poisoning training ----
    checks = {
        "basins": int(x.shape[0]),
        "days": int(x.shape[1]),
        "forcing_channels": int(x.shape[2]),
        "attributes": int(c.shape[1]),
        "basin_order_matches_gauge_ids": True,
        "x_nan": int(np.isnan(x).sum()),
        "y_nan": int(np.isnan(y).sum()),
        "c_nan": int(np.isnan(c).sum()),
        "forcing_sentinel_hits_converted_to_nan": int(sentinel_hits),
        "all_nan_forcing_basins": [],
        "all_nan_flow_basins": [],
    }
    for row, gauge_id in enumerate(gauge_ids):
        if np.isnan(x[row]).all():
            checks["all_nan_forcing_basins"].append(gauge_id)
        if np.isnan(y[row]).all():
            checks["all_nan_flow_basins"].append(gauge_id)
    if checks["all_nan_forcing_basins"]:
        raise RuntimeError(
            "Basins with no forcing at all: " + ", ".join(checks["all_nan_forcing_basins"][:10])
        )

    scaler_payload = fit_scaler(x, y, c, dates)
    scaler_payload["source"] = {"zip_md5": zip_md5, "gauge_ids_sha256": gauge_sha}

    # Arrays last: a half-written array must never look like a finished build.
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, array in (("x", x), ("y", y), ("c", c)):
        tmp = array_paths[name].with_name(array_paths[name].name + ".tmp")
        with open(tmp, "wb") as handle:
            np.save(handle, array)
        os.replace(tmp, array_paths[name])

    attrs_payload = {
        "forcing": list(FORCING_ORDER),
        "forcing_units": FORCING_UNITS,
        "attributes": list(ATTR_LST),
        "attribute_aliases": dict(ATTRIBUTE_ALIASES),
        "attribute_units_note": (
            "CAMELS v2.0 attribute table units; area_gages2 km2, p_mean mm/day"
        ),
        "categorical_encoding": {
            "method": "pd.factorize(sort=True); missing -> -1",
            "columns": sorted(encoded),
            "first_values": encoded,
        },
        "target": {
            "name": "usgsFlow",
            "units": "cfs",
            "note": "converted to m3/s in evaluation.py, never here",
        },
    }
    meta = {
        "dataset": {
            "title": CAMELS_DATASET_TITLE,
            "version": CAMELS_VERSION,
            "zenodo_record": CAMELS_ZENODO_RECORD,
            "doi": f"10.5281/zenodo.{CAMELS_ZENODO_RECORD}",
            "forcing": "Daymet",
        },
        "source": {"zip_md5": zip_md5},
        "time": {
            "start": DATASET_START,
            "end": DATASET_END,
            "n_days": len(dates),
            "calendar": "proleptic_gregorian",
        },
        "basins": {
            "n": n_basins,
            "gauge_ids_sha256": gauge_sha,
            "order": "data/gauge_ids.csv",
        },
        "arrays": {
            "x": {"shape": list(x.shape), "dtype": str(x.dtype),
                  "columns": list(FORCING_ORDER)},
            "y": {"shape": list(y.shape), "dtype": str(y.dtype),
                  "columns": ["usgsFlow_cfs"]},
            "c": {"shape": list(c.shape), "dtype": str(c.dtype),
                  "columns": list(ATTR_LST)},
        },
        "checks": checks,
        "split": {name: {"start": window[0], "end": window[1]}
                  for name, window in SPLIT_WINDOWS.items()},
        "context_days": CONTEXT_DAYS,
        "built_utc": _utc_now(),
    }

    _write_json_if_changed(PREPROCESSED_DIR / "attrs_order.json", attrs_payload)
    _write_json_if_changed(PREPROCESSED_DIR / "scaler.json", scaler_payload)
    _write_json_if_changed(meta_path, meta)

    print(f"  wrote    x{tuple(x.shape)} y{tuple(y.shape)} c{tuple(c.shape)}")
    print(f"           NaN: x={checks['x_nan']:,} y={checks['y_nan']:,} c={checks['c_nan']:,}"
          f" | sentinel -999 -> NaN: {sentinel_hits:,}")
    return meta


# ===========================================================================
# B2b — split spec, split pickles, equivalence gate
# ===========================================================================
#
# The split is fixed here, once, and everything downstream reads
# `data/split.json` rather than re-deriving dates:
#
#     train  WY1981-1995   1980-10-01 -> 1995-09-30   5 478 d
#     val    WY1996-1999   1995-10-01 -> 1999-09-30   1 461 d
#     test   WY2000-2014   1999-10-01 -> 2014-09-30   5 479 d
#
# The pickles keep the legacy tuple layout `(x, y, c)` so the existing loading
# code needs no format change.  Warm-up context is *not* baked into them: it is
# prepended at load time from the previous slice, which keeps every pickle a
# pure date window and keeps `y` of a slice exactly as long as its dates.

PICKLE_DIR = DATA_DIR / "pickles"
LEGACY_DIR = DATA_DIR / "legacy"
SPLIT_PATH = DATA_DIR / "split.json"
PICKLE_PROTOCOL = 4

# Day counts the plan promises; a mismatch means the date axis or the windows
# are wrong, never that the promise should be relaxed.
SPLIT_DAYS = {"train": 5478, "val": 1461, "test": 5479}

# The adjudicator is deliberately small and fixed: 5 basins spread across the
# 671-row order, 30 consecutive days well inside the scored val period (so no
# test target is touched before lock).
ADJUDICATION_BASINS = (0, 168, 336, 504, 670)
ADJUDICATION_START = "1997-01-01"
ADJUDICATION_DAYS = 30
ADJUDICATION_TOLERANCE = 1e-9

# Attribute columns that are strings in the published tables and become
# factorised integer codes in `c`; they cannot be compared to a numeric token.
CATEGORICAL_ATTRIBUTES = ("dom_land_cover", "geol_1st_class", "geol_2nd_class")


def _pickle_path(name):
    return PICKLE_DIR / f"{name}.pkl"


def _load_preprocessed():
    """Load the B2a arrays and their metadata, refusing to guess if absent."""
    paths = {name: PREPROCESSED_DIR / f"{name}.npy" for name in ("x", "y", "c")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "preprocessed arrays are missing: "
            + ", ".join(missing)
            + "\nrun first: python src/data_utils.py --build"
        )
    x = np.load(paths["x"], allow_pickle=False)
    y = np.load(paths["y"], allow_pickle=False)
    c = np.load(paths["c"], allow_pickle=False)
    meta = _read_json(PREPROCESSED_DIR / "meta.json", {}) or {}
    expected = {"x": (EXPECTED_BASINS, DATASET_DAYS, len(FORCING_ORDER)),
                "y": (EXPECTED_BASINS, DATASET_DAYS, 1),
                "c": (EXPECTED_BASINS, len(ATTR_LST))}
    for name, array in (("x", x), ("y", y), ("c", c)):
        if array.shape != expected[name]:
            raise RuntimeError(f"{name}.npy has shape {array.shape}, expected {expected[name]}")
        if array.dtype != np.float64:
            raise RuntimeError(f"{name}.npy has dtype {array.dtype}, expected float64")
    return x, y, c, meta


def split_offsets(dates=None):
    """Return ``(dates, offsets)`` for the 15/4/15 split.

    ``offsets[name]`` carries the inclusive ISO window, the half-open array
    indices and the day count.  The three windows must tile the record
    exactly: no gap, no overlap, and the promised day counts.
    """
    dates = _date_axis() if dates is None else dates
    offsets = {}
    cursor = 0
    for name in ("train", "val", "test"):
        start, end = SPLIT_WINDOWS[name]
        selected = np.flatnonzero(
            (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        )
        if len(selected) == 0:
            raise RuntimeError(f"{name} window {start}..{end} selects no days")
        first, last = int(selected[0]), int(selected[-1])
        if not np.array_equal(selected, np.arange(first, last + 1)):
            raise RuntimeError(f"{name} window {start}..{end} is not contiguous")
        if first != cursor:
            raise RuntimeError(
                f"{name} starts at day {first}, expected {cursor} (gap or overlap)"
            )
        actual_start = dates[first].strftime("%Y-%m-%d")
        actual_end = dates[last].strftime("%Y-%m-%d")
        if actual_start != start or actual_end != end:
            raise RuntimeError(
                f"{name} window requested {start}..{end}, date axis gives "
                f"{actual_start}..{actual_end}"
            )
        days = last + 1 - first
        if days != SPLIT_DAYS[name]:
            raise RuntimeError(f"{name} has {days} days, the plan promises {SPLIT_DAYS[name]}")
        offsets[name] = {
            "start": start,
            "end": end,
            "start_index": first,
            "stop_index": last + 1,
            "days": int(days),
        }
        cursor = last + 1
    if cursor != len(dates):
        raise RuntimeError(f"split covers {cursor} of {len(dates)} days")
    return dates, offsets


def _context_windows(offsets):
    """Warm-up context each scored slice borrows from the tail of the previous one."""
    windows = {"train": None}
    for name, previous in (("val", "train"), ("test", "val")):
        if offsets[previous]["days"] < CONTEXT_DAYS:
            raise RuntimeError(
                f"{previous} has {offsets[previous]['days']} days, "
                f"cannot lend {CONTEXT_DAYS} days of context to {name}"
            )
        start = offsets[name]["start_index"] - CONTEXT_DAYS
        windows[name] = {
            "source_slice": previous,
            "start_index": start,
            "stop_index": offsets[name]["start_index"],
            "days": CONTEXT_DAYS,
        }
    return windows


def _context_window_dates(dates, windows):
    """Attach ISO dates to the context windows (kept separate for readability)."""
    for name, window in windows.items():
        if window is None:
            continue
        window["start"] = dates[window["start_index"]].strftime("%Y-%m-%d")
        window["end"] = dates[window["stop_index"] - 1].strftime("%Y-%m-%d")
    return windows


def _split_sha256(spec):
    """Stable fingerprint of the split, quoted by every locked recipe."""
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_split_pickles(force=False):
    """Write ``data/pickles/{train,val,test}.pkl`` and the tracked ``data/split.json``.

    Idempotent: a second call reuses pickles whose SHA256 still matches the
    recorded one and does not rewrite ``split.json``.  ``force`` re-slices
    from the arrays regardless.
    """
    print("[B2b] split pickles: 15 / 4 / 15 water years")
    x, y, c, meta = _load_preprocessed()
    dates, offsets = split_offsets()
    context = _context_window_dates(dates, _context_windows(offsets))
    gauge_ids = load_gauge_ids(GAUGE_IDS_PATH, n_basins=x.shape[0])
    gauge_sha = _gauge_ids_sha256(gauge_ids)

    source = {
        "arrays": {name: list(array.shape)
                   for name, array in (("x", x), ("y", y), ("c", c))},
        "zip_md5": (meta.get("source") or {}).get("zip_md5"),
        "gauge_ids_sha256": gauge_sha,
    }
    previous = _read_json(SPLIT_PATH, {}) or {}
    reusable = (
        not force
        and previous.get("preprocessed") == source
        and isinstance(previous.get("slices"), dict)
        and all(_pickle_path(name).exists() for name in offsets)
    )

    slices = {}
    PICKLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, offset in offsets.items():
        path = _pickle_path(name)
        recorded = (previous.get("slices") or {}).get(name, {}) if reusable else {}
        if recorded.get("sha256") and _sha256_file(path) == recorded["sha256"]:
            entry = dict(recorded)
            print(f"  cached   {_rel(path)} ({_fmt_bytes(path.stat().st_size)})")
        else:
            start, stop = offset["start_index"], offset["stop_index"]
            payload = (x[:, start:stop, :], y[:, start:stop, :], c)
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "wb") as handle:
                pickle.dump(payload, handle, protocol=PICKLE_PROTOCOL)
            os.replace(tmp, path)
            entry = {"sha256": _sha256_file(path)}
            print(f"  wrote    {_rel(path)} ({_fmt_bytes(path.stat().st_size)})")
            del payload

        start, stop = offset["start_index"], offset["stop_index"]
        entry.update({
            "start": offset["start"],
            "end": offset["end"],
            "start_index": start,
            "stop_index": stop,
            "days": offset["days"],
            "y_nan": int(np.isnan(y[:, start:stop, :]).sum()),
            "pickle": _rel(path),
            "bytes": path.stat().st_size,
        })
        slices[name] = entry

    spec = {
        "protocol": {
            "scheme": "15 / 4 / 15 water years",
            "dates_are_inclusive": True,
            "pickle_format": "pickle tuple (x, y, c), float64, protocol 4",
            "warm_up": "context inputs prepended at load time, never stored",
        },
        "dataset": {
            "title": meta.get("dataset", {}).get("title", CAMELS_DATASET_TITLE),
            "version": meta.get("dataset", {}).get("version", CAMELS_VERSION),
            "zenodo_record": CAMELS_ZENODO_RECORD,
            "doi": f"10.5281/zenodo.{CAMELS_ZENODO_RECORD}",
            "time": meta.get("time"),
        },
        "preprocessed": source,
        "arrays": {
            "x": {"shape": list(x.shape), "dtype": str(x.dtype),
                  "columns": list(FORCING_ORDER)},
            "y": {"shape": list(y.shape), "dtype": str(y.dtype),
                  "columns": ["usgsFlow_cfs"]},
            "c": {"shape": list(c.shape), "dtype": str(c.dtype),
                  "columns": list(ATTR_LST)},
        },
        "slices": slices,
        "context": {
            "days": CONTEXT_DAYS,
            "rule": ("each scored slice borrows `days` of context *inputs* from the "
                     "tail of the previous slice; targets are never borrowed"),
            "windows": context,
        },
        "basins": {
            "n": int(x.shape[0]),
            "gauge_ids_sha256": gauge_sha,
            "order": _rel(GAUGE_IDS_PATH),
        },
    }
    spec["split_sha256"] = _split_sha256(spec)

    payload = dict(spec)
    for key in ("equivalence", "adjudication"):
        if previous.get(key):
            payload[key] = previous[key]
    changed = _write_json_if_changed(SPLIT_PATH, payload)

    print(f"  {'wrote   ' if changed else 'cached  '} {_rel(SPLIT_PATH)} "
          f"(split_sha256 {payload['split_sha256'][:12]}…)")
    for name, entry in slices.items():
        print(f"    {name:5s} {entry['start']} → {entry['end']}  "
              f"{entry['days']:5d} d  y NaN {entry['y_nan']:7,d}")
    return payload


def load_split(name, allow_test=False):
    """Load one pure slice as ``(x, y, c)``.

    The test slice is closed by default: only the evaluation entry point may
    pass ``allow_test=True``.  The guard lives here, next to the file, so no
    training module can reach the test targets by accident.
    """
    if name not in SPLIT_WINDOWS:
        raise ValueError(f"unknown split {name!r}; expected one of {sorted(SPLIT_WINDOWS)}")
    if name == "test" and not allow_test:
        raise PermissionError(
            "the test split is locked: only src/eval_test.py may open it "
            "(call load_split('test', allow_test=True))"
        )
    path = _pickle_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{_rel(path)} is missing\nrun first: python src/data_utils.py --split"
        )
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not (isinstance(payload, tuple) and len(payload) == 3):
        raise RuntimeError(f"{_rel(path)} is not a legacy (x, y, c) tuple")
    return payload


def load_split_with_context(name, ctx_days=CONTEXT_DAYS, allow_test=False):
    """Load a slice with warm-up context inputs prepended.

    The returned ``x`` has ``ctx_days + slice_days`` time steps; ``y`` and ``c``
    keep the scored slice's length/targets untouched, so the first scored day
    stays the first value of ``y``.
    """
    x, y, c = load_split(name, allow_test=allow_test)
    if name == "train" or ctx_days == 0:
        return x, y, c
    previous = "train" if name == "val" else "val"
    previous_x, _, _ = load_split(previous)
    if previous_x.shape[1] < ctx_days:
        raise RuntimeError(f"{previous} has fewer than {ctx_days} days of context")
    context = previous_x[:, -ctx_days:, :]
    return np.concatenate([context, x], axis=1), y, c


# ---------------------------------------------------------------------------
# B2b — legacy equivalence gate and independent adjudicator
# ---------------------------------------------------------------------------

def _compare_arrays(new, legacy, tolerance=1e-9):
    """NaN-aware comparison: identical masks, then max |difference| off-mask."""
    new_nan = np.isnan(new)
    legacy_nan = np.isnan(legacy)
    mismatch = int(np.count_nonzero(new_nan != legacy_nan))
    both = ~new_nan & ~legacy_nan
    if both.any():
        difference = np.abs(
            new[both].astype(np.float64) - legacy[both].astype(np.float64)
        )
        max_abs = float(difference.max())
    else:
        max_abs = 0.0
    return {
        "max_abs_diff": max_abs,
        "nan_mismatch": mismatch,
        "ok": bool(mismatch == 0 and max_abs <= tolerance),
    }


def _load_legacy_pickles():
    """Load the two pre-revision pickles kept in ``data/legacy/``."""
    paths = {
        "train": LEGACY_DIR / "training_file",
        "val": LEGACY_DIR / "validation_file",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "legacy pickles are missing: " + ", ".join(missing)
            + "\nthey are not tracked; restore them to data/legacy/ to re-run the gate"
        )
    loaded = {}
    for name, path in paths.items():
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if not (isinstance(payload, tuple) and len(payload) == 3):
            raise RuntimeError(f"{path} is not a legacy (x, y, c) tuple")
        loaded[name] = {"path": path, "arrays": payload}
    return loaded


def _published_attribute_tokens(gauge_ids, columns=None):
    """Numeric attribute values parsed from the published tables, token by token.

    This is the hand side of the adjudicator: ``csv`` module, raw text, and
    Python's correctly-rounded ``float()`` — nothing from ``read_attributes``
    or pandas is reused.  String columns are excluded because ``c`` stores
    their factorised codes, not their text.

    Returns ``(values, names)`` with ``values`` shaped ``(len(gauge_ids), len(names))``.
    """
    names = [name for name in (columns or ATTR_LST) if name not in CATEGORICAL_ATTRIBUTES]
    position = {name: index for index, name in enumerate(names)}
    row_of = {format_gauge_id(gauge_id): row for row, gauge_id in enumerate(gauge_ids)}
    values = np.full((len(gauge_ids), len(names)), np.nan, dtype=np.float64)
    found = set()

    for key in ("clim", "topo", "vege", "soil", "geol"):
        path = ATTRIBUTE_DIR / f"camels_{key}.txt"
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle, delimiter=";"))
        header = [cell.strip() for cell in rows[0]]
        gauge_column = header.index("gauge_id")
        sources = {}
        for name in names:
            source = ATTRIBUTE_ALIASES.get(name, name)
            if source in header:
                sources[name] = header.index(source)
        for line in rows[1:]:
            if len(line) <= gauge_column:
                continue
            row = row_of.get(format_gauge_id(line[gauge_column]))
            if row is None:
                continue
            for name, column in sources.items():
                token = line[column].strip()
                if token and token.lower() not in {"na", "nan", "null"}:
                    values[row, position[name]] = float(token)
                found.add(name)

    missing = [name for name in names if name not in found]
    if missing:
        raise RuntimeError(f"published attribute tables lack {missing}")
    return values, names


def verify_equivalence(tolerance=1e-9):
    """Compare the rebuild with the pre-revision pickles, raw first, scaled second.

    The raw comparison is the gate: identical NaN masks, ``y`` identical off
    the mask, ``x`` max |Δ| <= ``tolerance``.  ``c`` is held to a stronger
    test than the pickle: every numeric cell must equal the published table's
    token parsed with Python's ``float()``.  (The 2010-era pickle itself is
    1 ULP off in 156 of those cells — an old decimal parser, not a data
    difference — so comparing to it bit-for-bit would fail for the wrong
    reason.  The one-ULP count is reported, not hidden.)

    Any raw difference is a pipeline bug (channel order, units, row order,
    leap days) and returns False.  The scaled comparison is diagnostic: it
    shows whether the new scaler was fitted on the same window as the legacy
    one, and a constant ratio is recorded, not treated as failure.
    """
    x, y, c, meta = _load_preprocessed()
    dates, offsets = split_offsets()
    legacy = _load_legacy_pickles()

    # The tracked spec must still describe the trackers: a hand-edited
    # split.json would silently break every recipe that pins its hash.
    split_payload = _read_json(SPLIT_PATH, {}) or {}
    split_integrity = "absent"
    if split_payload.get("split_sha256"):
        spec = {key: split_payload[key] for key in
                ("protocol", "dataset", "preprocessed", "arrays", "slices", "context", "basins")}
        if _split_sha256(spec) != split_payload["split_sha256"]:
            raise RuntimeError(
                f"{_rel(SPLIT_PATH)} does not match its recorded split_sha256"
            )
        split_integrity = "ok"

    train_stop = offsets["train"]["stop_index"]
    # The legacy validation pickle runs to 2010-09-30, not to the new val end.
    legacy_stop = int(np.flatnonzero(dates == pd.Timestamp("2010-09-30"))[0]) + 1
    if legacy_stop - train_stop != legacy["val"]["arrays"][0].shape[1]:
        raise RuntimeError(
            f"legacy validation pickle covers {legacy['val']['arrays'][0].shape[1]} days, "
            f"the date axis says {legacy_stop - train_stop}"
        )

    windows = {
        "train": {
            "label": f"{dates[0]:%Y-%m-%d} → {dates[train_stop - 1]:%Y-%m-%d}",
            "new": (x[:, :train_stop, :], y[:, :train_stop, :]),
            "legacy": (legacy["train"]["arrays"][0], legacy["train"]["arrays"][1]),
        },
        "val": {
            "label": f"{dates[train_stop]:%Y-%m-%d} → {dates[legacy_stop - 1]:%Y-%m-%d}",
            "new": (x[:, train_stop:legacy_stop, :], y[:, train_stop:legacy_stop, :]),
            "legacy": (legacy["val"]["arrays"][0], legacy["val"]["arrays"][1]),
        },
    }

    print("\n== raw comparison (the gate) ==")
    raw = {}
    raw_ok = True
    for name, window in windows.items():
        legacy_x, legacy_y = window["legacy"]
        new_x, new_y = window["new"]
        if new_x.shape != legacy_x.shape or new_y.shape != legacy_y.shape:
            raise RuntimeError(
                f"{name}: shapes differ, new {new_x.shape}/{new_y.shape} "
                f"vs legacy {legacy_x.shape}/{legacy_y.shape}"
            )
        channels = {}
        for index, channel in enumerate(FORCING_ORDER):
            channels[channel] = _compare_arrays(
                new_x[:, :, index], legacy_x[:, :, index], tolerance
            )
        target = _compare_arrays(new_y, legacy_y, tolerance)
        raw[name] = {
            "window": window["label"],
            "days": int(new_x.shape[1]),
            "x": channels,
            "y": target,
        }
        raw_ok = raw_ok and all(item["ok"] for item in channels.values()) and target["ok"]
        print(f"  {name:5s} {window['label']}  {new_x.shape[1]:5d} d")
        for channel, item in channels.items():
            verdict = "PASS" if item["ok"] else "FAIL"
            print(f"    x.{channel:6s} max|Δ| {item['max_abs_diff']:.3e}  "
                  f"NaN mismatch {item['nan_mismatch']:6d}  {verdict}")
        verdict = "PASS" if target["ok"] else "FAIL"
        print(f"    y        max|Δ| {target['max_abs_diff']:.3e}  "
              f"NaN mismatch {target['nan_mismatch']:6d}  {verdict}")
        del new_x, new_y, legacy_x, legacy_y

    gauge_ids = load_gauge_ids(GAUGE_IDS_PATH, n_basins=c.shape[0])
    legacy_c = legacy["train"]["arrays"][2]
    attributes = _compare_arrays(c, legacy_c, tolerance)
    attributes["bit_identical"] = bool(np.array_equal(c, legacy_c, equal_nan=True))
    numeric, numeric_names = _published_attribute_tokens(gauge_ids)
    numeric_index = [ATTR_LST.index(name) for name in numeric_names]
    numeric_new = c[:, numeric_index]
    numeric_legacy = legacy_c[:, numeric_index]

    def _mismatch_count(left, right):
        both_nan = np.isnan(left) & np.isnan(right)
        return int(np.count_nonzero(~(left == right) & ~both_nan))

    new_mismatches = _mismatch_count(numeric_new, numeric)
    legacy_mismatches = _mismatch_count(numeric_legacy, numeric)
    ulp_cells = _mismatch_count(numeric_new, numeric_legacy)
    categorical_index = [ATTR_LST.index(name) for name in CATEGORICAL_ATTRIBUTES]
    categorical_same = bool(np.array_equal(
        c[:, categorical_index], legacy_c[:, categorical_index], equal_nan=True
    ))
    attributes["published_tokens"] = {
        "columns": numeric_names,
        "cells": int(numeric.size),
        "new_mismatches": new_mismatches,
        "legacy_mismatches": legacy_mismatches,
        "legacy_cells_differing_from_new": ulp_cells,
        "categorical_codes_identical": categorical_same,
    }
    attributes["ok"] = bool(
        attributes["nan_mismatch"] == 0
        and attributes["max_abs_diff"] <= tolerance
        and new_mismatches == 0
        and categorical_same
    )
    raw["c"] = attributes
    raw_ok = raw_ok and attributes["ok"]
    print(f"  c     35 columns   max|Δ| {attributes['max_abs_diff']:.3e}  "
          f"NaN mismatch {attributes['nan_mismatch']}  "
          f"{'PASS' if attributes['ok'] else 'FAIL'}")
    print(f"    numeric cells {numeric.size:,}: new matches published "
          f"{numeric.size - new_mismatches:,}/{numeric.size:,}, "
          f"legacy {numeric.size - legacy_mismatches:,}/{numeric.size:,} "
          f"(legacy 1-ULP cells: {ulp_cells:,}); "
          f"categorical codes {'identical' if categorical_same else 'DIFFER'}")

    print("\n== scaled comparison (diagnostic only) ==")
    scaled = {"status": "skipped"}
    try:
        from config import ExperimentConfig

        cfg = ExperimentConfig(
            data_opt=2,
            train_file=str(LEGACY_DIR / "training_file"),
            validation_file=str(LEGACY_DIR / "validation_file"),
        )
        legacy_scaled = load_and_scale_data(cfg)
        payload = _read_json(PREPROCESSED_DIR / "scaler.json", {}) or {}
        stats = dict(payload.get("attribute_statistics") or {})
        stats.update(payload.get("series_statistics") or {})
        legacy_stats = legacy_scaled["scaler"].stat_dict or {}
        missing_stats = [name for name in list(FORCING_ORDER) + ["runoff"] + ATTR_LST
                         if name not in stats or name not in legacy_stats]
        if missing_stats:
            raise RuntimeError(f"scaler statistics missing for {missing_stats[:5]}")

        legacy_x, _ = windows["val"]["legacy"]
        new_val_x = transform_by_statistics(
            legacy_x, list(FORCING_ORDER), stats, [], True
        )
        new_val_x[np.isnan(new_val_x)] = 0.0
        new_val_c = transform_by_statistics(c, ATTR_LST, stats, [], True)
        new_val_c[np.isnan(new_val_c)] = 0.0

        statistic_diffs = {}
        for name in list(FORCING_ORDER) + ["runoff"] + ATTR_LST:
            old, new = legacy_stats[name], stats[name]
            statistic_diffs[name] = {
                "legacy": [float(value) for value in old],
                "new": [float(value) for value in new],
                "mean_delta": float(new[2] - old[2]),
                "std_ratio": float(new[3] / old[3]) if old[3] else None,
            }
        scaled_x = _compare_arrays(new_val_x, legacy_scaled["val_x"], tolerance)
        scaled_c = _compare_arrays(new_val_c, legacy_scaled["val_c"], tolerance)
        worst_mean = max(statistic_diffs.values(), key=lambda row: abs(row["mean_delta"]))
        worst_ratio = max(
            (row for row in statistic_diffs.values() if row["std_ratio"] is not None),
            key=lambda row: abs(row["std_ratio"] - 1.0),
        )
        scaled = {
            "status": "compared",
            "statistics": statistic_diffs,
            "val_x": scaled_x,
            "val_c": scaled_c,
            "max_abs_mean_delta": float(abs(worst_mean["mean_delta"])),
            "std_ratio_range": [
                float(min(row["std_ratio"] for row in statistic_diffs.values()
                          if row["std_ratio"] is not None)),
                float(max(row["std_ratio"] for row in statistic_diffs.values()
                          if row["std_ratio"] is not None)),
            ],
            "ok": bool(scaled_x["ok"] and scaled_c["ok"]),
        }
        del legacy_scaled, new_val_x, new_val_c
        print(f"  val_x scaled  max|Δ| {scaled_x['max_abs_diff']:.3e}  "
              f"NaN mismatch {scaled_x['nan_mismatch']}")
        print(f"  val_c scaled  max|Δ| {scaled_c['max_abs_diff']:.3e}  "
              f"NaN mismatch {scaled_c['nan_mismatch']}")
        print(f"  statistics    max |mean Δ| {scaled['max_abs_mean_delta']:.3e}  "
              f"std ratio range {scaled['std_ratio_range'][0]:.6f} – "
              f"{scaled['std_ratio_range'][1]:.6f}")
        print(f"  (worst std ratio: {worst_ratio['std_ratio']:.6f})")
    except Exception as error:  # noqa: BLE001 - diagnostics must not mask the gate
        scaled = {"status": "skipped", "reason": f"{type(error).__name__}: {error}"}
        print(f"  skipped: {scaled['reason']}")
        print("  (raw comparison above still decides pass/fail)")

    block = {
        "checked_utc": _utc_now(),
        "tolerance": tolerance,
        "split_integrity": split_integrity,
        "legacy_window": {
            "train": windows["train"]["label"],
            "val": windows["val"]["label"],
        },
        "legacy_files": [
            {
                "path": _rel(window["path"]),
                "sha256": _sha256_file(window["path"]),
                "bytes": window["path"].stat().st_size,
            }
            for window in legacy.values()
        ],
        "raw": raw,
        "raw_ok": bool(raw_ok),
        "scaled": scaled,
    }

    _record_split_check("equivalence", block)
    verdict = "GREEN" if raw_ok else "RED"
    print(f"\n  raw equivalence: {verdict}")
    return bool(raw_ok)


def _record_split_check(key, block):
    """Store one verification block in ``data/split.json`` without touching the spec."""
    payload = _read_json(SPLIT_PATH, {}) or {}
    if not payload:
        print(f"  note: {_rel(SPLIT_PATH)} does not exist; {key} result not recorded")
        return
    payload[key] = block
    _write_json_if_changed(SPLIT_PATH, payload)


def _hand_text_frame(path, names, skiprows=0):
    """Read a whitespace CAMELS text file with an independent reader."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, skiprows=skiprows, names=names)
    stamp = pd.to_datetime(
        {"year": frame["year"], "month": frame["month"], "day": frame["day"]}
    )
    return frame, stamp


def adjudicate():
    """Two-of-three tiebreaker: hand-derived vs new pipeline vs legacy pickle.

    Five basins x 30 days, read straight from the published text files with
    separate readers (pandas for the time series, ``csv`` + ``float()`` for
    the attributes; no ``build_arrays`` helper is called).  Forcing channels,
    streamflow, the documented basin-normalised runoff and all numeric
    attributes are compared against both the rebuild and the 2010-era pickle
    at the same basin rows and days.
    """
    x, y, c, _ = _load_preprocessed()
    dates, offsets = split_offsets()
    gauge_ids = load_gauge_ids(GAUGE_IDS_PATH, n_basins=x.shape[0])

    start = pd.Timestamp(ADJUDICATION_START)
    selected = np.flatnonzero(dates == start)
    if len(selected) != 1:
        raise RuntimeError(f"{ADJUDICATION_START} is not on the date axis")
    day_index = int(selected[0])
    stop = day_index + ADJUDICATION_DAYS
    if not (offsets["val"]["start_index"] <= day_index
            and stop <= offsets["val"]["stop_index"]):
        raise RuntimeError("adjudication window must sit inside the val slice")

    legacy = _load_legacy_pickles()
    legacy_val_x, legacy_val_y, _ = legacy["val"]["arrays"]
    legacy_offset = day_index - offsets["val"]["start_index"]

    info = pd.read_csv(
        EXTRACT_DIR / METADATA_ROOT / "gauge_information.txt",
        sep="\t", header=None, skiprows=1, dtype=str,
    ).iloc[:, :6]
    info.columns = ["huc", "gauge_id", "gauge_name", "lat", "lon", "area"]
    huc_by_gauge = {
        str(gauge).strip().zfill(8): str(huc).strip().zfill(2)
        for gauge, huc in zip(info["gauge_id"], info["huc"])
    }

    clim = pd.read_csv(ATTRIBUTE_DIR / "camels_clim.txt", sep=";",
                       dtype={"gauge_id": str}).set_index("gauge_id")
    topo = pd.read_csv(ATTRIBUTE_DIR / "camels_topo.txt", sep=";",
                       dtype={"gauge_id": str}).set_index("gauge_id")
    clim.index = clim.index.astype(str).str.zfill(8)
    topo.index = topo.index.astype(str).str.zfill(8)

    forcing_names = ["year", "month", "day", "hr", "dayl", "prcp", "srad",
                     "swe", "tmax", "tmin", "vp"]
    flow_names = ["gauge", "year", "month", "day", "q", "flag"]
    window_label = f"{start:%Y-%m-%d} → {dates[stop - 1]:%Y-%m-%d}"

    adjudication_ids = [format_gauge_id(gauge_ids[row]) for row in ADJUDICATION_BASINS]
    published_values, published_names = _published_attribute_tokens(adjudication_ids)
    categorical_index = [ATTR_LST.index(name) for name in CATEGORICAL_ATTRIBUTES]

    print(f"\n== adjudication: {len(ADJUDICATION_BASINS)} basins × "
          f"{ADJUDICATION_DAYS} days, {window_label} ==")
    results = []
    passed = True
    for position, row in enumerate(ADJUDICATION_BASINS):
        gauge_id = format_gauge_id(gauge_ids[row])
        huc = huc_by_gauge.get(gauge_id)
        if huc is None:
            raise RuntimeError(f"{gauge_id} is missing from gauge_information.txt")

        forcing_path = (EXTRACT_DIR / FORCING_ROOT / huc
                        / f"{gauge_id}_lump_cida_forcing_leap.txt")
        flow_path = EXTRACT_DIR / FLOW_ROOT / huc / f"{gauge_id}_streamflow_qc.txt"
        forcing_frame, forcing_stamp = _hand_text_frame(forcing_path, forcing_names, skiprows=4)
        forcing_mask = (forcing_stamp >= start) & (forcing_stamp < start + pd.Timedelta(days=ADJUDICATION_DAYS))
        forcing_window = forcing_frame.loc[forcing_mask]
        if len(forcing_window) != ADJUDICATION_DAYS:
            raise RuntimeError(f"{gauge_id}: hand forcing has {len(forcing_window)} days, "
                               f"expected {ADJUDICATION_DAYS}")
        hand_forcing = np.column_stack([
            forcing_window["dayl"].to_numpy(dtype=np.float64),
            forcing_window["prcp"].to_numpy(dtype=np.float64),
            forcing_window["srad"].to_numpy(dtype=np.float64),
            0.5 * (forcing_window["tmax"].to_numpy(dtype=np.float64)
                   + forcing_window["tmin"].to_numpy(dtype=np.float64)),
            forcing_window["vp"].to_numpy(dtype=np.float64),
        ])
        hand_forcing[hand_forcing == FORCING_SENTINEL] = np.nan

        flow_frame, flow_stamp = _hand_text_frame(flow_path, flow_names)
        flow_mask = (flow_stamp >= start) & (flow_stamp < start + pd.Timedelta(days=ADJUDICATION_DAYS))
        hand_flow = flow_frame.loc[flow_mask, "q"].to_numpy(dtype=np.float64, copy=True)
        if len(hand_flow) != ADJUDICATION_DAYS:
            raise RuntimeError(f"{gauge_id}: hand streamflow has {len(hand_flow)} days, "
                               f"expected {ADJUDICATION_DAYS}")
        hand_flow[hand_flow < 0] = np.nan

        # Documented basin normalisation, written out rather than calling
        # basin_norm(): cfs -> m3/s -> m/day -> fraction of mean precipitation.
        hand_area = float(topo.at[gauge_id, "area_gages2"])
        hand_p_mean = float(clim.at[gauge_id, "p_mean"])
        conversion = 0.0283168 * 3600.0 * 24.0
        hand_runoff = (hand_flow * conversion) / (hand_area * 1e6 * hand_p_mean * 1e-3)

        new_forcing = x[row, day_index:stop, :]
        new_flow = y[row, day_index:stop, 0]
        new_runoff = (new_flow * conversion) / (hand_area * 1e6 * hand_p_mean * 1e-3)
        old_forcing = legacy_val_x[row, legacy_offset:legacy_offset + ADJUDICATION_DAYS, :]
        old_flow = legacy_val_y[row, legacy_offset:legacy_offset + ADJUDICATION_DAYS, 0]
        old_runoff = (old_flow * conversion) / (hand_area * 1e6 * hand_p_mean * 1e-3)

        comparisons = {}
        for index, channel in enumerate(FORCING_ORDER):
            comparisons[f"x.{channel}"] = {
                "hand_vs_new": _compare_arrays(hand_forcing[:, index], new_forcing[:, index],
                                               ADJUDICATION_TOLERANCE),
                "hand_vs_legacy": _compare_arrays(hand_forcing[:, index], old_forcing[:, index],
                                                  ADJUDICATION_TOLERANCE),
            }
        comparisons["y"] = {
            "hand_vs_new": _compare_arrays(hand_flow, new_flow, ADJUDICATION_TOLERANCE),
            "hand_vs_legacy": _compare_arrays(hand_flow, old_flow, ADJUDICATION_TOLERANCE),
        }
        comparisons["runoff"] = {
            "hand_vs_new": _compare_arrays(hand_runoff, new_runoff, ADJUDICATION_TOLERANCE),
            "hand_vs_legacy": _compare_arrays(hand_runoff, old_runoff, ADJUDICATION_TOLERANCE),
        }
        legacy_c_row = legacy["train"]["arrays"][2][row]
        for index, name in enumerate(published_names):
            column = ATTR_LST.index(name)
            comparisons[f"c.{name}"] = {
                "hand_vs_new": _compare_arrays(
                    np.array([published_values[position, index]]),
                    np.array([c[row, column]]), 0.0,
                ),
                "hand_vs_legacy": _compare_arrays(
                    np.array([published_values[position, index]]),
                    np.array([legacy_c_row[column]]), 0.0,
                ),
            }
        categorical_same = bool(np.array_equal(
            c[row, categorical_index], legacy_c_row[categorical_index], equal_nan=True
        ))

        new_ok = all(item["hand_vs_new"]["ok"] for item in comparisons.values())
        legacy_ok = all(item["hand_vs_legacy"]["ok"] for item in comparisons.values())
        legacy_cells_off = [key for key, item in comparisons.items()
                            if key.startswith("c.") and not item["hand_vs_legacy"]["ok"]]
        if new_ok and legacy_ok:
            verdict = "hand = new = legacy"
        elif new_ok:
            verdict = "hand = new ≠ legacy (new wins)"
        elif legacy_ok:
            verdict = "hand = legacy ≠ new (legacy wins)"
        else:
            verdict = "hand differs from both"
        passed = passed and new_ok and categorical_same

        forcing_new = max(item["hand_vs_new"]["max_abs_diff"]
                          for key, item in comparisons.items() if key.startswith("x."))
        forcing_old = max(item["hand_vs_legacy"]["max_abs_diff"]
                          for key, item in comparisons.items() if key.startswith("x."))
        attribute_new = max(item["hand_vs_new"]["max_abs_diff"]
                            for key, item in comparisons.items() if key.startswith("c."))
        attribute_old = max(item["hand_vs_legacy"]["max_abs_diff"]
                            for key, item in comparisons.items() if key.startswith("c."))
        print(f"  row {row:3d}  {gauge_id}  HUC {huc}  {verdict}")
        print(f"    forcing max|Δ|  new {forcing_new:.3e}  legacy {forcing_old:.3e}")
        print(f"    y       max|Δ|  new {comparisons['y']['hand_vs_new']['max_abs_diff']:.3e}  "
              f"legacy {comparisons['y']['hand_vs_legacy']['max_abs_diff']:.3e}")
        print(f"    runoff  max|Δ|  new {comparisons['runoff']['hand_vs_new']['max_abs_diff']:.3e}  "
              f"legacy {comparisons['runoff']['hand_vs_legacy']['max_abs_diff']:.3e}")
        print(f"    attrs   max|Δ|  new {attribute_new:.3e}  legacy {attribute_old:.3e}"
              + (f"  ({len(legacy_cells_off)} 1-ULP legacy cells)" if legacy_cells_off else ""))

        results.append({
            "row": row,
            "gauge_id": gauge_id,
            "huc": huc,
            "verdict": verdict,
            "new_ok": bool(new_ok),
            "legacy_ok": bool(legacy_ok),
            "legacy_cells_off": legacy_cells_off,
            "categorical_codes_identical": categorical_same,
            "comparisons": comparisons,
        })

    new_agree = sum(1 for entry in results if entry["new_ok"])
    legacy_agree = sum(1 for entry in results if entry["legacy_ok"])
    block = {
        "checked_utc": _utc_now(),
        "window": {"start": start.strftime("%Y-%m-%d"),
                   "end": dates[stop - 1].strftime("%Y-%m-%d"),
                   "days": ADJUDICATION_DAYS},
        "tolerance": ADJUDICATION_TOLERANCE,
        "basins": [entry["gauge_id"] for entry in results],
        "new_agrees": new_agree,
        "legacy_agrees": legacy_agree,
        "results": results,
        "passed": bool(passed),
    }
    if legacy_agree < len(results):
        block["explanation"] = (
            "the rebuild matches the hand-derived published tokens exactly; the "
            "2010-era pickle differs by one float64 ULP in the listed attribute "
            "cells (an old decimal-parser artefact). Two-of-three: hand and new agree."
        )
    _record_split_check("adjudication", block)
    print(f"\n  adjudication: hand = new in {new_agree}/{len(results)} basins; "
          f"legacy agrees in {legacy_agree}/{len(results)}"
          + ("" if passed else " — the new pipeline disagrees with the hand path: investigate"))
    return bool(passed)


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------

def _update_manifest(downloads, layout, extraction, meta):
    """Merge download + build provenance into the tracked ``data/manifest.json``."""
    previous = _read_json(MANIFEST_PATH, {}) or {}
    payload = {
        "dataset": {
            "title": CAMELS_DATASET_TITLE,
            "version": CAMELS_VERSION,
            "zenodo_record": CAMELS_ZENODO_RECORD,
            "doi": f"10.5281/zenodo.{CAMELS_ZENODO_RECORD}",
            "landing_page": "https://ral.ucar.edu/dataset/"
                            "camels-catchment-attributes-and-meteorology-large-sample-studies-dataset-downloads",
            "checksum_policy": (
                "MD5 is the checksum Zenodo publishes and is verified on every "
                "build; SHA256 is computed locally for provenance."
            ),
        },
        "retrieved_utc": previous.get("retrieved_utc") or _utc_now(),
        "files": downloads,
        "skipped": {
            "reason": "not model inputs; re-downloadable from the same record",
            "keys": [
                "basin_timeseries_v1p2_modelOutput_daymet.zip",
                "basin_timeseries_v1p2_modelOutput_maurer.zip",
                "basin_timeseries_v1p2_modelOutput_nldas.zip",
                "basin_set_full_res.zip",
                "pet_harg.zip",
            ],
        },
        "extraction": {
            "local_path": _rel(EXTRACT_DIR),
            "members": extraction.get("n_members"),
            "bytes": extraction.get("bytes"),
            "layout": _rel(LAYOUT_PATH),
            "coverage": layout.get("coverage"),
        },
        "build": {
            "arrays": meta["arrays"],
            "time": meta["time"],
            "basins": meta["basins"],
            "checks": meta["checks"],
            "split": meta["split"],
            "context_days": meta["context_days"],
            "preprocessed": {
                "x": _rel(PREPROCESSED_DIR / "x.npy"),
                "y": _rel(PREPROCESSED_DIR / "y.npy"),
                "c": _rel(PREPROCESSED_DIR / "c.npy"),
                "scaler": _rel(PREPROCESSED_DIR / "scaler.json"),
                "attrs_order": _rel(PREPROCESSED_DIR / "attrs_order.json"),
                "meta": _rel(PREPROCESSED_DIR / "meta.json"),
            },
            "built_utc": meta.get("built_utc"),
        },
    }
    changed = _write_json_if_changed(MANIFEST_PATH, payload)
    print(f"[5/5] manifest {'updated' if changed else 'unchanged'}: {_rel(MANIFEST_PATH)}")
    return payload


def build_all(force=False, force_download=False):
    """Run the whole B2a pipeline: download → layout → extract → arrays → manifest."""
    started = time.time()
    downloads = download_raw(force=force_download)
    layout = probe_layout(force=force)
    extraction = extract_raw(force=force)
    meta = build_arrays(force=force)
    _update_manifest(downloads, layout, extraction, meta)
    checks = meta["checks"]
    print()
    print(f"B2a complete in {time.time() - started:.1f} s")
    print(f"  x {tuple(meta['arrays']['x']['shape'])}  "
          f"y {tuple(meta['arrays']['y']['shape'])}  "
          f"c {tuple(meta['arrays']['c']['shape'])}  "
          f"({meta['time']['start']} → {meta['time']['end']})")
    print(f"  basins {checks['basins']} | days {checks['days']} | "
          f"NaN y {checks['y_nan']:,} | NaN c {checks['c_nan']:,}")
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Native CAMELS v1.2 acquisition + preprocessing (B2a) and the "
            "15/4/15 split, equivalence gate and adjudicator (B2b)."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build",
        action="store_true",
        help="run the acquisition pipeline (default when no mode flag is given)",
    )
    mode.add_argument(
        "--split",
        action="store_true",
        help="write data/pickles/{train,val,test}.pkl and data/split.json",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="compare the rebuild with the legacy pickles; non-zero only on a raw failure",
    )
    mode.add_argument(
        "--adjudicate",
        action="store_true",
        help="5 basins x 30 days against a hand-derived path; non-zero on disagreement",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild extraction/arrays/pickles even when checksums match",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download the archive and attribute tables even when checksums match",
    )
    args = parser.parse_args(argv)

    if args.verify:
        return 0 if verify_equivalence() else 1
    if args.adjudicate:
        return 0 if adjudicate() else 1
    if args.split:
        write_split_pickles(force=args.force)
        return 0
    build_all(force=args.force, force_download=args.force_download)
    return 0


if __name__ == "__main__":
    sys.exit(main())
