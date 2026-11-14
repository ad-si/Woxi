#!/usr/bin/env python3
"""Build resources/country_indicators.tsv.gz, the latest values of the
further CountryData statistics (birth rate, life expectancy, poverty, …).

Source: World Bank Open Data, World Development Indicators
(https://data.worldbank.org), licensed CC BY 4.0
(https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets).

Input: scripts/country_data_entities.tsv (see gen_country_data.py).

Output: a header line naming the Wolfram Language properties, then one
line per country:
    entity  value  value  …
Each value is the latest available year of the property, converted to the
scale the Wolfram Language uses (fractions instead of percentages or
per-mille rates), or empty when the World Bank has no figure.
The units are attached by src/functions/country_data.rs.

Usage: uv run scripts/gen_country_indicators.py
"""

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTITIES = ROOT / "scripts" / "country_data_entities.tsv"
OUT = ROOT / "resources" / "country_indicators.tsv.gz"
FIRST_YEAR, LAST_YEAR = 1990, 2024
API = "https://api.worldbank.org/v2/country/all/indicator/{}?format=json" \
      "&per_page=30000&date={}:{}"

# (property, indicator, factor): value = indicator * factor
DIRECT = [
    ("ChildPopulation", "SP.POP.0014.TO", 1),
    ("AdultPopulation", "SP.POP.1564.TO", 1),
    ("ElderlyPopulation", "SP.POP.65UP.TO", 1),
    ("FemalePopulation", "SP.POP.TOTL.FE.IN", 1),
    ("FemaleChildPopulation", "SP.POP.0014.FE.IN", 1),
    ("FemaleAdultPopulation", "SP.POP.1564.FE.IN", 1),
    ("FemaleElderlyPopulation", "SP.POP.65UP.FE.IN", 1),
    ("MalePopulation", "SP.POP.TOTL.MA.IN", 1),
    ("MaleChildPopulation", "SP.POP.0014.MA.IN", 1),
    ("MaleAdultPopulation", "SP.POP.1564.MA.IN", 1),
    ("MaleElderlyPopulation", "SP.POP.65UP.MA.IN", 1),
    ("PopulationGrowth", "SP.POP.GROW", 0.01),
    ("BirthRateFraction", "SP.DYN.CBRT.IN", 0.001),
    ("DeathRateFraction", "SP.DYN.CDRT.IN", 0.001),
    ("TotalFertilityRate", "SP.DYN.TFRT.IN", 1),
    ("LifeExpectancy", "SP.DYN.LE00.IN", 1),
    ("FemaleLifeExpectancy", "SP.DYN.LE00.FE.IN", 1),
    ("MaleLifeExpectancy", "SP.DYN.LE00.MA.IN", 1),
    ("InfantMortalityFraction", "SP.DYN.IMRT.IN", 0.001),
    ("FemaleInfantMortalityFraction", "SP.DYN.IMRT.FE.IN", 0.001),
    ("MaleInfantMortalityFraction", "SP.DYN.IMRT.MA.IN", 0.001),
    ("HIVAIDSFraction", "SH.DYN.AIDS.ZS", 0.01),
    ("LiteracyFraction", "SE.ADT.LITR.ZS", 0.01),
    ("FemaleLiteracyFraction", "SE.ADT.LITR.FE.ZS", 0.01),
    ("MaleLiteracyFraction", "SE.ADT.LITR.MA.ZS", 0.01),
    ("LaborForce", "SL.TLF.TOTL.IN", 1),
    ("UnemploymentFraction", "SL.UEM.TOTL.ZS", 0.01),
    ("PovertyFraction", "SI.POV.NAHC", 0.01),
    ("CurrentAccountBalance", "BN.CAB.XOKA.CD", 1),
    ("EconomicAid", "DT.ODA.ODAT.CD", 1),
    ("ExternalDebt", "DT.DOD.DECT.CD", 1),
    ("ForeignExchangeReserves", "FI.RES.TOTL.CD", 1),
    ("GDPAtParity", "NY.GDP.MKTP.PP.CD", 1),
    ("GDPRealGrowth", "NY.GDP.MKTP.KD.ZG", 0.01),
    ("GiniIndex", "SI.POV.GINI", 0.01),
    ("IndustrialProductionGrowth", "NV.IND.TOTL.KD.ZG", 0.01),
    ("InflationRate", "FP.CPI.TOTL.ZG", 0.01),
    ("ManufacturingValueAdded", "NV.IND.MANF.CD", 1),
    ("NationalIncome", "NY.GNP.MKTP.CD", 1),
    ("AgriculturalValueAdded", "NV.AGR.TOTL.CD", 1),
    ("IndustrialValueAdded", "NV.IND.TOTL.CD", 1),
    ("TotalConsumption", "NE.CON.TOTL.CD", 1),
    ("HouseholdConsumption", "NE.CON.PRVT.CD", 1),
    ("GovernmentConsumption", "NE.CON.GOVT.CD", 1),
    ("GrossInvestment", "NE.GDI.TOTL.CD", 1),
    ("FixedInvestment", "NE.GDI.FTOT.CD", 1),
    ("ExportValue", "NE.EXP.GNFS.CD", 1),
    ("ImportValue", "NE.IMP.GNFS.CD", 1),
    ("InventoryChange", "NE.GDI.STKB.CD", 1),
    ("RailwayLength", "IS.RRS.TOTL.KM", 1),
    ("PhoneLines", "IT.MLT.MAIN", 1),
    ("CellularPhones", "IT.CEL.SETS", 1),
]

# (property, indicator, factor): value = indicator * factor * population
# of the same year
PER_POPULATION = [
    ("AnnualBirths", "SP.DYN.CBRT.IN", 0.001),
    ("AnnualDeaths", "SP.DYN.CDRT.IN", 0.001),
    ("InternetUsers", "IT.NET.USER.ZS", 0.01),
]

# (property, indicator): value = indicator / population of the same year
OVER_POPULATION = [
    ("MigrationRateFraction", "SM.POP.NETM"),
]


def indicator(code):
    """{iso3: {year: value}} for one World Development Indicator."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(
                    API.format(code, FIRST_YEAR, LAST_YEAR), timeout=600) as r:
                data = json.load(r)
            break
        except OSError:
            time.sleep(3 + 3 * attempt)
    else:
        raise RuntimeError(code)
    out = {}
    for row in data[1] or []:
        if row["value"] is not None and row["countryiso3code"]:
            out.setdefault(row["countryiso3code"], {})[int(row["date"])] = \
                row["value"]
    return out


def latest(series):
    return series[max(series)] if series else None


def num(v):
    if v is None:
        return ""
    if float(v).is_integer():
        return str(int(v))
    # Five significant digits: the precision the estimates carry.
    return f"{v:.5g}"


def main():
    # The API answers slowly, so fetch every indicator at once.
    codes = {"SP.POP.TOTL"} | {c for _, c, *_ in
                               DIRECT + PER_POPULATION + OVER_POPULATION}
    with ThreadPoolExecutor(max_workers=16) as pool:
        fetched = dict(zip(codes, pool.map(indicator, codes)))
    pop = fetched["SP.POP.TOTL"]
    columns = []  # (property, {iso3: value})
    for prop, code, factor in DIRECT:
        data = fetched[code]
        columns.append((prop, {
            iso3: latest(s) * factor for iso3, s in data.items() if s
        }))
    for prop, code, factor in PER_POPULATION:
        data = fetched[code]
        col = {}
        for iso3, s in data.items():
            years = [y for y in s if y in pop.get(iso3, {})]
            if years:
                y = max(years)
                col[iso3] = round(s[y] * factor * pop[iso3][y])
        columns.append((prop, col))
    for prop, code in OVER_POPULATION:
        data = fetched[code]
        col = {}
        for iso3, s in data.items():
            years = [y for y in s if pop.get(iso3, {}).get(y)]
            if years:
                y = max(years)
                col[iso3] = s[y] / pop[iso3][y]
        columns.append((prop, col))
    lines = [
        "# Source: World Bank World Development Indicators, CC BY 4.0. "
        f"Latest year {FIRST_YEAR}-{LAST_YEAR}. "
        "Regenerate with scripts/gen_country_indicators.py",
        "\t".join(["entity"] + [p for p, _ in columns]),
    ]
    for line in ENTITIES.read_text().splitlines():
        entity, iso3 = line.split("\t")[:2]
        lines.append("\t".join(
            [entity] + [num(col.get(iso3)) for _, col in columns]))
    with open(OUT, "wb") as raw, gzip.GzipFile(
            fileobj=raw, mode="wb", compresslevel=9, mtime=0) as f:
        f.write(("\n".join(lines) + "\n").encode())


if __name__ == "__main__":
    main()
