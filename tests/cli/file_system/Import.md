# `Import`

`Import[path, format]` reads a file and parses it, while
`Export[path, expr, format]` writes an expression to a file.
Both route through the same format back-ends as `ImportString` and
`ExportString`, and accept the same options.

Common recognised formats include `"CSV"`, `"TSV"`, `"JSON"`, `"Text"`,
`"Lines"`, `"PNG"`, `"SVG"`, and `"String"`.

### Common options

- **`"CharacterEncoding"`** — text encoding (default `"UTF8"`).
- **`"HeaderLines"`** — number of leading lines to treat as header.
- **`"Delimiter"`** — separator character for CSV/TSV.
- **`"Numeric"`** — if `True`, numeric-looking fields are parsed as numbers.
- **`"IncludeWindowsLineBreaks"`** — for text-based writers.


### CERN ROOT files (Woxi extension)

Woxi can import [CERN ROOT](https://root.cern/) files — the standard
container format of particle physics — which the official Wolfram Language
cannot. `Import["file.root"]` (or `Import[path, "ROOT"]` for files without
the `.root` extension) walks the file's directory structure and returns an
Association mapping each stored object's name to a decoded value:

- **`TObjString`** — the contained string.
- **`TH1C` / `TH1S` / `TH1I` / `TH1F` / `TH1D` histograms** — an Association
    with `"NBins"`, `"XMin"`, `"XMax"`, `"Entries"`, `"BinContents"`,
    `"Underflow"`, and `"Overflow"` (plus `"BinEdges"` for variable-width
    binning).
- **`TH2C` / `TH2S` / `TH2I` / `TH2F` / `TH2D` histograms** — an Association
    with `"NBinsX"`, `"XMin"`, `"XMax"`, `"NBinsY"`, `"YMin"`, `"YMax"`,
    `"Entries"`, and `"BinContents"` as an `NBinsX × NBinsY` matrix (row *i*
    holds x-bin *i* over all y-bins; the underflow/overflow border cells are
    dropped; `"XBinEdges"`/`"YBinEdges"` appear for variable-width binning).
- **`TTree`** — an Association with `"Entries"` and `"Branches"`
    (branch name → leaf specification or streamed class).
- **`TDirectory`** — a nested Association of the directory's contents.
- Any other class — an Association with `"ClassName"` and `"Title"` so the
    object is at least visible.

Uncompressed, zlib-, and LZ4-compressed records are supported.

```wolfram
data = Import["experiment.root"]
(* <|hist -> <|ClassName -> TH1D, Title -> , NBins -> 4, XMin -> 1.,
     XMax -> 5., Entries -> 9., BinContents -> {2., 2., 2., 3.},
     Underflow -> 0., Overflow -> 0.|>,
     events -> <|ClassName -> TTree, Title -> , Entries -> 5,
     Branches -> <|x -> x/D, n -> n/L|>|>|> *)

data["hist"]["BinContents"]
(* {2., 2., 2., 3.} *)
```

#### Reading TTree data

The directory walk lists trees as metadata only, so importing a file with a
multi-gigabyte tree stays cheap. Branch *columns* are read through element
paths (objects in subdirectories are addressed as `"dir/subdir/object"`):

```wolfram
Import["experiment.root", {"ROOT", "events", "x"}]
(* {1.1, 2.2, 3.3, 4.4, 5.5} *)

Import["experiment.root", {"ROOT", "events", {"x", "n"}}]
(* <|x -> {1.1, 2.2, 3.3, 4.4, 5.5}, n -> {10, 20, 30, 40, 50}|> *)

Import["experiment.root", {"ROOT", "events", "Data"}]
(* every branch: <|x -> {…}, n -> {…}|> — prefer selecting branches
   on large trees *)

Import["experiment.root", {"ROOT", "events"}]
(* the tree's metadata Association, same as the directory walk *)
```

For files with the `.root` extension the leading `"ROOT"` element is
optional: `Import["experiment.root", {"events", "x"}]`.

Decoded branch layouts (all cross-verified against uproot):

- flat basic types (`TLeafB/S/I/L/F/D/O`, signed and unsigned) and
    fixed-size arrays,
- variable-length (leaf-count) arrays — one list per entry,
- `std::vector<double>`, `std::vector<int>`, and the other basic-type
    vectors — one list per entry,
- `TLorentzVector` — `<|"Px" -> …, "Py" -> …, "Pz" -> …, "E" -> …|>`
    per entry.

Branches with unsupported layouts (e.g. split objects or nested vectors)
report a descriptive error in place of their data while the rest of the
tree still decodes.

Since wolframscript has no ROOT importer, these examples are not part of the
conformance test suite.
