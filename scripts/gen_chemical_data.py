#!/usr/bin/env python3
"""Build resources/chemical_data.tsv.gz, the offline table behind ChemicalData.

Source: PubChem (https://pubchem.ncbi.nlm.nih.gov), NCBI/NLM. PubChem data is
in the public domain (https://www.ncbi.nlm.nih.gov/home/about/policies/).
Every structural value (formula, weight, SMILES, IUPAC name, atom order,
bonds, 2-D depiction coordinates, 3-D conformer coordinates) is downloaded
from the PUG REST service by PubChem compound ID.

Input: scripts/chemical_data_entities.tsv with the columns
    entity  cid  name
where `entity` is the Wolfram Language entity name (e.g. "AceticAcid"),
`cid` the PubChem compound ID, and `name` the lowercase common name
ChemicalData reports as "Name". Only identifiers come from that list.

Output columns (tab separated, one compound per line):
    entity cid name formula weight smiles iupac atoms charges bonds xy xyz pka
  atoms   element symbols in PubChem's atom order, comma separated
  charges formal charges as "index:charge" pairs, comma separated
  bonds   "a-b-order" triples (1-based, order 1/2/3), comma separated
  xy      2-D depiction coordinates, "x y" pairs, comma separated
  xyz     3-D conformer coordinates (angstrom), "x y z" triples, comma
          separated; empty when PubChem has no conformer
  pka     first acid dissociation constant stated as "pKa = x" in PubChem's
          "Dissociation Constants" section; empty when there is none

Usage: uv run scripts/gen_chemical_data.py
"""

import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTITIES = ROOT / "scripts" / "chemical_data_entities.tsv"
OUT = ROOT / "resources" / "chemical_data.tsv.gz"
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"


def fetch(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 + attempt * 2)
        except urllib.error.URLError:
            time.sleep(2 + attempt * 2)
    raise RuntimeError(f"failed: {url}")


def parse_sdf(text):
    """Atoms, charges, bonds and coordinates of a V2000 molfile."""
    lines = text.splitlines()
    counts = lines[3]
    n_atoms, n_bonds = int(counts[0:3]), int(counts[3:6])
    atoms, coords, charges = [], [], {}
    for line in lines[4:4 + n_atoms]:
        coords.append(tuple(float(line[i:i + 10]) for i in (0, 10, 20)))
        atoms.append(line[31:34].strip())
    bonds = []
    for line in lines[4 + n_atoms:4 + n_atoms + n_bonds]:
        bonds.append((int(line[0:3]), int(line[3:6]), int(line[6:9])))
    for line in lines[4 + n_atoms + n_bonds:]:
        if line.startswith("M  CHG"):
            parts = line.split()[3:]
            for i in range(0, len(parts), 2):
                charges[int(parts[i])] = int(parts[i + 1])
        if line.startswith("M  END"):
            break
    return atoms, charges, bonds, coords


PKA = re.compile(
    r"\bpKa1?\s*(?:\([^)]*\))?\s*(?:=|:|of|is|value of)\s*(-?\d+(?:\.\d+)?)")


def first_pka(cid):
    """The first acid dissociation constant PubChem's "Dissociation
    Constants" section states as "pKa = x", or "" when there is none."""
    text = fetch("https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/"
                 f"compound/{cid}/JSON?heading=Dissociation+Constants")
    if not text:
        return ""
    for value in re.findall(r'"String": "([^"]*)"', text):
        m = PKA.search(value)
        if m:
            return m.group(1)
    return ""


def fmt(x):
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def main():
    rows = [
        line.split("\t")
        for line in ENTITIES.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    out = []
    for entity, cid, name in rows:
        props = json.loads(fetch(
            f"{BASE}/{cid}/property/MolecularFormula,MolecularWeight,"
            "IsomericSMILES,IUPACName/JSON"))["PropertyTable"]["Properties"][0]
        sdf2 = fetch(f"{BASE}/{cid}/record/SDF?record_type=2d")
        sdf3 = fetch(f"{BASE}/{cid}/record/SDF?record_type=3d")
        atoms, charges, bonds, xy = parse_sdf(sdf2)
        xyz = parse_sdf(sdf3)[3] if sdf3 else []
        smiles = props.get("IsomericSMILES") or props.get("SMILES", "")
        out.append("\t".join([
            entity, cid, name,
            props["MolecularFormula"], str(props["MolecularWeight"]),
            smiles, props.get("IUPACName", ""),
            ",".join(atoms),
            ",".join(f"{i}:{c}" for i, c in sorted(charges.items())),
            ",".join(f"{a}-{b}-{o}" for a, b, o in bonds),
            ",".join(f"{fmt(x)} {fmt(y)}" for x, y, _ in xy),
            ",".join(" ".join(fmt(v) for v in p) for p in xyz),
            first_pka(cid),
        ]))
        print(entity, file=sys.stderr)
        time.sleep(0.25)  # PubChem asks for at most 5 requests per second
    text = ("# Source: PubChem (NCBI/NLM), public domain. "
            "Regenerate with scripts/gen_chemical_data.py\n"
            + "\n".join(out) + "\n")
    # mtime=0 keeps the archive byte-identical across regenerations.
    with open(OUT, "wb") as raw, gzip.GzipFile(
            fileobj=raw, mode="wb", compresslevel=9, mtime=0) as f:
        f.write(text.encode())


if __name__ == "__main__":
    main()
