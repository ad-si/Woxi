#!/usr/bin/env python3
"""Build resources/cities.tsv.gz, the world-cities table behind CityData.

Source: GeoNames (https://www.geonames.org), licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Files: cities15000.zip,
admin1CodesASCII.txt and countryInfo.txt from
https://download.geonames.org/export/dump/.

Keeps the cities with at least MIN_POPULATION people. Names are written the
way the Wolfram Language spells city entities: the ASCII name with its
spaces and punctuation removed (`{"NewYork", "NewYork", "UnitedStates"}`),
the country under its WL entity name (scripts/country_data_entities.tsv).

Output columns (tab separated, one city per line):
    name region country label latitude longitude population
  label is the display name, left empty when it is the name itself

Usage: uv run scripts/gen_city_data.py
"""

import gzip
import io
import re
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "resources" / "cities.tsv.gz"
COUNTRIES = ROOT / "scripts" / "country_data_entities.tsv"
BASE = "https://download.geonames.org/export/dump/"
MIN_POPULATION = 100_000


def fetch(name):
    with urllib.request.urlopen(BASE + name, timeout=300) as r:
        return r.read()


def camel(s):
    words = re.split(r"[^A-Za-z0-9]+", s)
    return "".join(w[:1].upper() + w[1:] for w in words if w)


def main():
    iso3_to_entity = {}
    for line in COUNTRIES.read_text().splitlines():
        f = line.split("\t")
        iso3_to_entity[f[1]] = f[0]
    iso2_to_entity = {}
    for line in fetch("countryInfo.txt").decode().splitlines():
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if f[1] in iso3_to_entity:
            iso2_to_entity[f[0]] = iso3_to_entity[f[1]]
    admin1 = {}
    for line in fetch("admin1CodesASCII.txt").decode().splitlines():
        f = line.split("\t")
        admin1[f[0]] = f[2]
    raw = zipfile.ZipFile(io.BytesIO(fetch("cities15000.zip")))
    rows = []
    for line in raw.read("cities15000.txt").decode().splitlines():
        f = line.split("\t")
        population = int(f[14] or 0)
        country = iso2_to_entity.get(f[8])
        if population < MIN_POPULATION or not country:
            continue
        region = camel(admin1.get(f"{f[8]}.{f[10]}", ""))
        rows.append((country, -population, "\t".join([
            camel(f[2]), region, country,
            "" if f[1] == camel(f[2]) else f[1],
            f"{float(f[4]):.4f}", f"{float(f[5]):.4f}", str(population)])))
    rows.sort()
    # Keep the largest city of each identity (two "São José" in one state
    # collapse to the same entity name), and drop cities with no region.
    seen, unique = set(), []
    for row in rows:
        key = tuple(row[2].split("\t")[:3])
        if key in seen or not key[1]:
            continue
        seen.add(key)
        unique.append(row)
    rows = unique
    body = ("# Source: GeoNames cities15000 (CC BY 4.0), cities with at least "
            f"{MIN_POPULATION} people. "
            "Regenerate with scripts/gen_city_data.py\n"
            + "\n".join(r for *_, r in rows) + "\n")
    with open(OUT, "wb") as out, gzip.GzipFile(
            fileobj=out, mode="wb", compresslevel=9, mtime=0) as g:
        g.write(body.encode())


if __name__ == "__main__":
    main()
