"""
Downloads the PDEBench data files the experiments need into this repository's data/ tree, with md5
verification. The file list (name, official URL, relative path, md5) is copied from PDEBench's own
download index (`pdebench/data_download/pdebench_data_urls.csv` in the PDEBench repository); the
files are hosted on DaRUS, PDEBench's public data repository.

  advection  5 files, 1D_Advection_Sols_beta{0.1,0.4,1.0,2.0,7.0}.hdf5  (~8.2 GB each, ~41 GB)
  reacdiff  16 files, ReacDiff_Nu{0.5,1.0,2.0,5.0}_Rho{1.0,2.0,5.0,10.0}.hdf5  (~4.1 GB each, ~66 GB)

Files already present with the right md5 are skipped; downloads go through a temporary .part file.

Usage: python pde/download_pdebench_subset.py [advection] [reacdiff]
       (no argument = both groups; a file name from the list above downloads just that file)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
URL = "https://darus.uni-stuttgart.de/api/access/datafile/{}"

ADVECTION_PATH = "1D/Advection/Train/"
REACDIFF_PATH = "1D/ReactionDiffusion/Train/"

# (file name, DaRUS datafile id, relative path, md5)
FILES = {
    "advection": [
        ("1D_Advection_Sols_beta0.1.hdf5", 255672, ADVECTION_PATH, "b4be2fc3383f737c76033073e6d2ccfb"),
        ("1D_Advection_Sols_beta0.4.hdf5", 255674, ADVECTION_PATH, "d595bbfd2c659df995a93cd40d6ea568"),
        ("1D_Advection_Sols_beta1.0.hdf5", 255675, ADVECTION_PATH, "1fe41923a4123db55bf4e89bea32e142"),
        ("1D_Advection_Sols_beta2.0.hdf5", 255677, ADVECTION_PATH, "0e9beff98693089c38e213a1f2b9ac43"),
        ("1D_Advection_Sols_beta7.0.hdf5", 255664, ADVECTION_PATH, "72a286ad2fcba574bd20bc045ba82d58"),
    ],
    "reacdiff": [
        ("ReacDiff_Nu0.5_Rho1.0.hdf5", 133177, REACDIFF_PATH, "69a429239778d529cd419ed5888ea835"),
        ("ReacDiff_Nu0.5_Rho10.0.hdf5", 133178, REACDIFF_PATH, "ff7c724b18e7ebe02e19c179852f48ee"),
        ("ReacDiff_Nu0.5_Rho2.0.hdf5", 133179, REACDIFF_PATH, "ac907daa7e483d203a5c77567cdea561"),
        ("ReacDiff_Nu0.5_Rho5.0.hdf5", 133180, REACDIFF_PATH, "fb149e7540d8977af158bb8fec1048a3"),
        ("ReacDiff_Nu1.0_Rho1.0.hdf5", 133181, REACDIFF_PATH, "bd73c2f3448d03e95e98c3831fc8fa70"),
        ("ReacDiff_Nu1.0_Rho10.0.hdf5", 133182, REACDIFF_PATH, "a94e65631881a27ddae3ef74caf53093"),
        ("ReacDiff_Nu1.0_Rho2.0.hdf5", 133183, REACDIFF_PATH, "112c01a76447162bd67c8c1073f58ca2"),
        ("ReacDiff_Nu1.0_Rho5.0.hdf5", 133184, REACDIFF_PATH, "fa224c9d143de37ac6914d391e70f425"),
        ("ReacDiff_Nu2.0_Rho1.0.hdf5", 133185, REACDIFF_PATH, "925a7e2c9b9e40ad2dd44b7002ec882b"),
        ("ReacDiff_Nu2.0_Rho10.0.hdf5", 133186, REACDIFF_PATH, "a3e25a9fb8a99f010352ad3dd1afd596"),
        ("ReacDiff_Nu2.0_Rho2.0.hdf5", 133187, REACDIFF_PATH, "426353b241acfbc64067acb1bdc80ade"),
        ("ReacDiff_Nu2.0_Rho5.0.hdf5", 133188, REACDIFF_PATH, "f2890bdac5103a3b78fac5f80e57b760"),
        ("ReacDiff_Nu5.0_Rho1.0.hdf5", 133189, REACDIFF_PATH, "0ed75b55f61bec11c47d379e34959e54"),
        ("ReacDiff_Nu5.0_Rho10.0.hdf5", 133190, REACDIFF_PATH, "264c80af0e2a6cc1f70e87275e0f6ac4"),
        ("ReacDiff_Nu5.0_Rho2.0.hdf5", 133191, REACDIFF_PATH, "f927f5d85f2e9bd97dff31b4dab051ae"),
        ("ReacDiff_Nu5.0_Rho5.0.hdf5", 133192, REACDIFF_PATH, "c9c26e2b5f2d4bf5bcbbd4ba34fefd5e"),
    ],
}


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(name: str, datafile_id: int, rel_path: str, md5: str) -> None:
    target = DATA_ROOT / rel_path / name
    if target.exists():
        if md5sum(target) == md5:
            print(f"[skip] {name} already present, md5 ok")
            return
        print(f"[redo] {name} exists but md5 mismatch, re-downloading")
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    url = URL.format(datafile_id)
    print(f"[get ] {name} <- {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        done = 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
                done += len(chunk)
                if done % (1 << 28) < (1 << 22):  # progress roughly every 256MB
                    print(f"       ... {done / 1e9:.1f} GB", flush=True)
    got = md5sum(part)
    if got != md5:
        part.unlink()
        raise RuntimeError(f"{name}: md5 mismatch ({got} != {md5})")
    part.rename(target)
    print(f"[done] {name} ({target.stat().st_size / 1e9:.1f} GB), md5 ok")


def main() -> None:
    by_name = {row[0]: row for rows in FILES.values() for row in rows}
    selection = sys.argv[1:] or list(FILES)
    rows = []
    for item in selection:
        if item in FILES:
            rows += FILES[item]
        elif item in by_name:
            rows.append(by_name[item])
        else:
            raise SystemExit(f"unknown group or file {item!r}; groups: {list(FILES)}, files: {sorted(by_name)}")
    for row in rows:
        download_one(*row)


if __name__ == "__main__":
    main()
