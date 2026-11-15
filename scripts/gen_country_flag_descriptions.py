#!/usr/bin/env python3
"""Build resources/country_flag_descriptions.tsv.gz, the texts behind
CountryData[country, "FlagDescription"].

Sources:
  - Flag descriptions: the CIA World Factbook (public domain), via the
    JSON mirror at https://github.com/factbook/factbook.json, whose files
    are named by FIPS 10-4 code and grouped by region.
  - ISO 3166-1 alpha-3 -> FIPS 10-4 codes:
    https://github.com/datasets/country-codes (ODC-PDDL; only used to look
    up the codes, not shipped).

Input: scripts/country_data_entities.tsv (entity and iso3 columns).

Output columns (tab separated, one country per line):
    entity  description
  description  the "description:" part of the Factbook's "Flag" entry,
               as plain text (HTML tags and entities removed)

Usage: uv run scripts/gen_country_flag_descriptions.py
"""

import csv
import gzip
import html
import io
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTITIES = ROOT / "scripts" / "country_data_entities.tsv"
OUT = ROOT / "resources" / "country_flag_descriptions.tsv.gz"
CODES_CSV = (
    "https://raw.githubusercontent.com/datasets/country-codes/"
    "main/data/country-codes.csv"
)
FACTBOOK_URL = (
    "https://raw.githubusercontent.com/factbook/factbook.json/master/{}/{}.json"
)
REGIONS = [
    "africa",
    "antarctica",
    "australia-oceania",
    "central-america-n-caribbean",
    "central-asia",
    "east-n-southeast-asia",
    "europe",
    "middle-east",
    "north-america",
    "south-america",
    "south-asia",
]

# Entities whose FIPS code the code table lacks.
EXTRA = {"Kosovo": "kv", "GazaStrip": "gz", "WestBank": "we"}


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read()
    except urllib.error.HTTPError:
        return None


def description(fips: str) -> str | None:
    for region in REGIONS:
        data = fetch(FACTBOOK_URL.format(region, fips))
        if data is None:
            continue
        flag = json.loads(data).get("Government", {}).get("Flag", {})
        text = flag.get("text", "")
        # "<strong>description:</strong> … <br><br><strong>history:</strong> …"
        match = re.search(
            r"<strong>description:</strong>(.*?)(?:<br>|$)", text, re.S
        )
        body = match.group(1) if match else text
        body = html.unescape(re.sub(r"<[^>]+>", "", body))
        body = " ".join(body.split())
        return body or None
    return None


def main() -> None:
    rows = csv.DictReader(io.StringIO(fetch(CODES_CSV).decode("utf-8")))
    # A territory with several FIPS codes lists them run together
    # (Serbia "RIKV"); its first code is the main one.
    fips_of = {
        r["ISO3166-1-Alpha-3"]: r["FIPS"].strip().lower()[:2]
        for r in rows
        if len(r["FIPS"].strip()) >= 2
    }

    entities = []
    for line in ENTITIES.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        entity, iso3 = line.split("\t")[:2]
        code = EXTRA.get(entity) or fips_of.get(iso3)
        if code is None:
            print(f"no FIPS code for {entity}")
            continue
        entities.append((entity, code))

    with ThreadPoolExecutor(16) as pool:
        texts = list(pool.map(lambda e: description(e[1]), entities))

    lines = ["# Flag descriptions: CIA World Factbook (public domain). "
             "Regenerate with scripts/gen_country_flag_descriptions.py"]
    for (entity, code), text in zip(entities, texts):
        if text is None:
            print(f"no Factbook flag description for {entity} ({code})")
            continue
        lines.append(f"{entity}\t{text}")

    with gzip.open(OUT, "wt", compresslevel=9) as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 1} flag descriptions to {OUT}")


if __name__ == "__main__":
    main()
