#!/usr/bin/env python3
"""Build resources/country_flags.tsv.gz, the flag images behind
CountryData[country, "Flag"].

Sources:
  - Flag images: https://flagcdn.com (160 px wide PNGs rendered from the
    Wikimedia Commons flag SVGs, which are in the public domain).
  - ISO 3166-1 alpha-3 -> alpha-2 codes:
    https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes
    (CC BY-SA 4.0; only used to look up the codes, not shipped).

Input: scripts/country_data_entities.tsv (entity and iso3 columns).

Output columns (tab separated, one country per line):
    entity  png
  png  the flag as a base64-encoded PNG

Usage: uv run scripts/gen_country_flags.py
"""

import base64
import csv
import gzip
import io
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTITIES = ROOT / "scripts" / "country_data_entities.tsv"
OUT = ROOT / "resources" / "country_flags.tsv.gz"
ISO_CSV = (
    "https://raw.githubusercontent.com/lukes/"
    "ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
)
FLAG_URL = "https://flagcdn.com/w160/{}.png"

# Entities without an ISO 3166-1 alpha-3 code of their own.
EXTRA = {"Kosovo": "xk", "GazaStrip": "ps", "WestBank": "ps"}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def main() -> None:
    rows = csv.DictReader(io.StringIO(fetch(ISO_CSV).decode("utf-8")))
    alpha2 = {r["alpha-3"]: r["alpha-2"].lower() for r in rows}

    cache: dict[str, str] = {}
    lines = ["# Flags: flagcdn.com renderings of the public-domain Wikimedia "
             "Commons flags. Regenerate with scripts/gen_country_flags.py"]
    for line in ENTITIES.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        entity, iso3 = line.split("\t")[:2]
        code = EXTRA.get(entity) or alpha2.get(iso3)
        if code is None:
            print(f"no ISO code for {entity}")
            continue
        if code not in cache:
            cache[code] = base64.b64encode(fetch(FLAG_URL.format(code))) \
                .decode("ascii")
        lines.append(f"{entity}\t{cache[code]}")

    with gzip.open(OUT, "wt", compresslevel=9) as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 1} flags to {OUT}")


if __name__ == "__main__":
    main()
