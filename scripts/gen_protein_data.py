#!/usr/bin/env python3
"""Build resources/protein_data.tsv.gz, the offline table behind ProteinData.

Sources:
  - UniProtKB (https://www.uniprot.org), CC BY 4.0: the reviewed human
    entries, their canonical amino acid sequence and their Gene Ontology
    annotations.
  - Gene Ontology (http://geneontology.org), CC BY 4.0: the GO term names,
    as UniProt reports them next to each annotation.
  - HGNC (https://www.genenames.org), CC0: the approved gene symbol, which
    ProteinData uses as the protein's standard name, and the approved name,
    which ProteinData reports as "Name".

The slice holds every reviewed human protein annotated with one of the GO
terms listed in CLASSES below or with one of their descendant terms, so
that `ProteinData[class]` (the proteins annotated directly with that term)
lists the whole class for each of them.

Output (tab separated):
    P  symbol  name  sequence  functions  processes  components
        one protein per line; the last three columns are comma separated
        GO IDs (molecular functions, biological processes, cellular
        components)
    T  go_id  term_name
        one GO term per line

Usage: uv run scripts/gen_protein_data.py
"""

import gzip
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "resources" / "protein_data.tsv.gz"

# GO terms whose whole membership (and that of their descendants) is bundled.
CLASSES = ["GO:0003779"]  # actin binding

UNIPROT = "https://rest.uniprot.org/uniprotkb/stream"
HGNC = "https://rest.genenames.org/fetch/symbol/"


def fetch(url, headers=None):
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode()
        except OSError:  # HTTP, URL and dropped-connection errors
            time.sleep(2 + attempt * 2)
    raise RuntimeError(f"failed: {url}")


TERM = re.compile(r"(.*?) \[(GO:\d{7})\]")


def go_terms(field, names):
    ids = []
    for part in field.split("; "):
        m = TERM.fullmatch(part.strip())
        if m:
            names[m.group(2)] = m.group(1)
            ids.append(m.group(2))
    return ids


def main():
    names = {}
    proteins = {}
    for go in CLASSES:
        query = f"organism_id:9606 AND reviewed:true AND go:{go[3:]}"
        url = UNIPROT + "?" + urllib.parse.urlencode(
            {
                "query": query,
                "fields": "gene_primary,sequence,go_f,go_p,go_c",
                "format": "tsv",
            }
        )
        rows = fetch(url).splitlines()[1:]
        for row in rows:
            symbol, seq, gof, gop, goc = (row.split("\t") + [""] * 5)[:5]
            if not symbol or symbol in proteins:
                continue
            f = go_terms(gof, names)
            p = go_terms(gop, names)
            c = go_terms(goc, names)
            proteins[symbol] = [seq, f, p, c]

    lines = []
    for symbol in sorted(proteins):
        seq, f, p, c = proteins[symbol]
        doc = json.loads(
            fetch(HGNC + urllib.parse.quote(symbol), {"Accept": "application/json"})
        )["response"]["docs"]
        name = doc[0]["name"] if doc else symbol
        time.sleep(0.1)
        lines.append(
            "\t".join(
                ["P", symbol, name, seq, ",".join(f), ",".join(p), ",".join(c)]
            )
        )
    used = {t for v in proteins.values() for ids in v[1:] for t in ids}
    for go in sorted(used):
        lines.append(f"T\t{go}\t{names[go]}")

    with gzip.open(OUT, "wt", compresslevel=9) as out:
        out.write("# ProteinData slice: UniProtKB (CC BY 4.0), GO (CC BY 4.0), "
                  "HGNC (CC0). Regenerate with scripts/gen_protein_data.py\n")
        out.write("\n".join(lines) + "\n")
    print(f"{len(proteins)} proteins, {len(used)} terms -> {OUT}")


if __name__ == "__main__":
    main()
