#!/usr/bin/env python3
"""Build resources/country_data.tsv.gz, the statistics table behind
CountryData.

Source: World Bank Open Data, World Development Indicators
(https://data.worldbank.org), licensed CC BY 4.0
(https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets).
Indicators: SP.POP.TOTL (population), NY.GDP.MKTP.CD (GDP, current US$),
AG.SRF.TOTL.K2 (surface area, km^2), AG.LND.TOTL.K2 (land area, km^2).

Input: scripts/country_data_entities.tsv with the columns
    entity  iso3  name  fullname  continents
naming each country the way the Wolfram Language does (identifiers and
names only; no statistics). `continents` lists the continent classes the
country belongs to, its primary continent first.

Output columns (tab separated, one country per line):
    entity iso3 name fullname continents area landarea population gdp
  area        latest surface area in km^2, or empty
  landarea    latest land area in km^2, or empty
  population  yearly values FIRST_YEAR..LAST_YEAR, comma separated, empty
              entries for missing years
  gdp         likewise, in current US dollars

Usage: uv run scripts/gen_country_data.py
"""

import gzip
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTITIES = ROOT / "scripts" / "country_data_entities.tsv"
OUT = ROOT / "resources" / "country_data.tsv.gz"
FIRST_YEAR, LAST_YEAR = 1960, 2024
API = "https://api.worldbank.org/v2/country/all/indicator/{}?format=json" \
      "&per_page=30000&date={}:{}"


def indicator(code):
    """{iso3: {year: value}} for one World Development Indicator."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(
                    API.format(code, FIRST_YEAR, LAST_YEAR), timeout=120) as r:
                data = json.load(r)
            break
        except OSError:
            time.sleep(3 + 3 * attempt)
    else:
        raise RuntimeError(code)
    out = {}
    for row in data[1]:
        if row["value"] is not None and row["countryiso3code"]:
            out.setdefault(row["countryiso3code"], {})[int(row["date"])] = \
                row["value"]
    return out


def num(v):
    if v is None:
        return ""
    if float(v).is_integer():
        return str(int(v))
    # Seven significant digits: the precision the estimates carry.
    return f"{v:.7g}"


def main():
    pop = indicator("SP.POP.TOTL")
    gdp = indicator("NY.GDP.MKTP.CD")
    area = indicator("AG.SRF.TOTL.K2")
    land = indicator("AG.LND.TOTL.K2")
    years = range(FIRST_YEAR, LAST_YEAR + 1)
    lines = [
        "# Source: World Bank World Development Indicators, CC BY 4.0. "
        f"Years {FIRST_YEAR}-{LAST_YEAR}. "
        "Regenerate with scripts/gen_country_data.py"
    ]
    for line in ENTITIES.read_text().splitlines():
        entity, iso3, name, fullname, continents = line.split("\t")
        a = area.get(iso3, {})
        latest_area = a[max(a)] if a else None
        la = land.get(iso3, {})
        latest_land = la[max(la)] if la else None
        lines.append("\t".join([
            entity, iso3, name, fullname, continents, num(latest_area),
            num(latest_land),
            ",".join(num(pop.get(iso3, {}).get(y)) for y in years),
            ",".join(num(gdp.get(iso3, {}).get(y)) for y in years),
        ]))
    with open(OUT, "wb") as raw, gzip.GzipFile(
            fileobj=raw, mode="wb", compresslevel=9, mtime=0) as f:
        f.write(("\n".join(lines) + "\n").encode())


if __name__ == "__main__":
    main()
