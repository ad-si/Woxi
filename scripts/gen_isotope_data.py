#!/usr/bin/env python3
"""Build resources/isotope_data.tsv.gz, the nuclide chart behind IsotopeData.

Source: NUBASE2020 (F.G. Kondev, M. Wang, W.J. Huang, S. Naimi and G. Audi,
"The NUBASE2020 evaluation of nuclear physics properties", Chinese Physics C
45 (2021) 030001, open access under CC BY 3.0), as distributed by the IAEA
Atomic Mass Data Center: https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt

Only ground states (isomer index 0) of nuclides with Z >= 1 are kept.

Output columns (tab separated, one nuclide per line):
    Z  A  mass_excess_keV  half_life  unit  decay_modes
  half_life   the value as printed by NUBASE ("stbl" for stable, empty when
              unknown); a trailing "#" marks a value from systematics and
              is dropped
  unit        NUBASE's unit (ys ... s m h d y ky My Gy ... Yy)
  decay_modes NUBASE's decay field without the isotopic abundance entry
              ("A=100;SF=5.44e-5 7")

Usage: uv run scripts/gen_isotope_data.py
"""

import gzip
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "resources" / "isotope_data.tsv.gz"
URL = "https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt"


def main():
    # The IAEA server rejects Python's default User-Agent.
    request = urllib.request.Request(URL, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(request, timeout=120) as r:
        text = r.read().decode("latin-1")
    rows = []
    for line in text.splitlines():
        if line.startswith("#") or len(line) < 20:
            continue
        a = int(line[0:3])
        z = int(line[4:7])
        isomer = line[7]
        if isomer != "0" or z == 0:
            continue
        mass_excess = line[18:31].strip().replace("#", "")
        half_life = line[69:78].strip().replace("#", "")
        unit = line[78:80].strip()
        modes = line[119:209].strip() if len(line) > 119 else ""
        modes = ";".join(m for m in modes.split(";")
                         if m and not m.startswith("IS="))
        rows.append("\t".join(
            [str(z), str(a), mass_excess, half_life, unit, modes]))
    body = ("# Source: NUBASE2020 (Kondev et al., Chin. Phys. C 45 030001), "
            "CC BY 3.0, via IAEA AMDC. "
            "Regenerate with scripts/gen_isotope_data.py\n"
            + "\n".join(rows) + "\n")
    with open(OUT, "wb") as raw, gzip.GzipFile(
            fileobj=raw, mode="wb", compresslevel=9, mtime=0) as f:
        f.write(body.encode())


if __name__ == "__main__":
    main()
