#!/usr/bin/env wolframscript
(* Regenerates resources/example_data_meta.tsv.gz, the metadata table behind
   Woxi's generated ExampleData.

   Woxi does not bundle Wolfram's example data. It generates stand-in data
   whose *structure* matches: the catalogue, the property names, the head and
   shape of every value, and summary statistics (dimensions, counts, sample
   rates, column types, value ranges, ...). This script records exactly that
   structure from a wolframscript installation, and never the data itself:

   - exact values only for metadata properties (sizes, counts, names, types);
   - a skeleton (`sk`) for everything else: atoms are replaced by their head,
     homogeneous lists by `Rep[n, {elementSkeletons}]`, numeric arrays by
     `Arr[dims, heads]`, images/audio/graphs/meshes by their key metadata;
   - `@`-columns with summary statistics (min/max/mean/sd of numeric
     columns, per-channel image means, audio RMS, degree statistics, ...).

   Long text (descriptions, sources, books) is never recorded, only its
   length: Woxi writes its own.

   Usage (serially; wolframscript must not run concurrently):

     wolframscript -file scripts/example_data_meta.wls /tmp/example_data_meta.tsv \
       [collection ...]
     gzip -9n < /tmp/example_data_meta.tsv > resources/example_data_meta.tsv.gz

   The table goes to the file named first (the kernel prints download
   progress on stdout).

   Without arguments every collection is captured; this downloads every
   dataset from the Wolfram server once (a few GB, mostly NetworkGraph and
   LinearProgramming); Matrix entries are recorded from their index only.

   Output format: for each collection a header line
     @Collection<TAB>column<TAB>column...
   followed by one line per entry
     name<TAB>cell<TAB>cell...
   where a cell is "=" followed by the InputForm of the exact value, the
   InputForm of a skeleton, or empty when the entry lacks the property.
   "@"-columns hold exact InputForm values without the "=". *)

$HistoryLength = 0;

fmt[x_] := StringReplace[
  ToString[x, InputForm, CharacterEncoding -> "ASCII", PageWidth -> Infinity],
  {"\t" -> "\\t", "\n" -> "\\n", "\r" -> "\\r"}];

round[x_?NumericQ] := With[{r = N[x]}, If[r == 0, 0., Round[r, 10.^(Floor[Log10[Abs[r]]] - 3)]]];
round[x_] := x;

(* ---- skeleton of a value: structure without data ---- *)
sk[x_Integer] := Integer;
sk[x_Real] := Real;
sk[x_Rational] := Rational;
sk[x_Complex] := Complex;
sk[x_String] := String;
sk[x_Symbol] := x;
sk[x_Image] := Sk["Image", ImageDimensions[x], ImageChannels[x], ImageType[x],
  ImageColorSpace[x], Lookup[Options[x], Interleaving, None]];
sk[x_Image3D] := Sk["Image3D", ImageDimensions[x], ImageChannels[x], ImageType[x],
  ImageColorSpace[x]];
sk[x_Audio] := Sk["Audio", AudioSampleRate[x], AudioChannels[x], AudioLength[x],
  AudioType[x]];
sk[x_NumericArray] := Sk["NumericArray", Dimensions[x], NumericArrayType[x]];
sk[x_SparseArray] := Sk["SparseArray", Dimensions[x], Length[x["NonzeroValues"]],
  x["Background"], Union[Head /@ x["NonzeroValues"]]];
sk[x_Graph] := Sk["Graph", VertexCount[x], EdgeCount[x], DirectedGraphQ[x],
  WeightedGraphQ[x], Union[sk /@ VertexList[x]]];
sk[x_MeshRegion] := Sk["MeshRegion", MeshCellCount[x], RegionEmbeddingDimension[x]];
sk[x_BoundaryMeshRegion] := Sk["BoundaryMeshRegion", MeshCellCount[x],
  RegionEmbeddingDimension[x]];
sk[x_Dataset] := Sk["Dataset", sk[Normal[x]]];
sk[x_Quantity] := Sk["Quantity", sk[QuantityMagnitude[x]], QuantityUnit[x]];
sk[x_DateObject] := Sk["DateObject", Length[First[x]], x[[2]]];
sk[x_Missing] := x;
sk[x_List] /; ArrayQ[x, _, NumericQ] && Length[x] > 0 :=
  Arr[Dimensions[x], Union[Head /@ Flatten[x]]];
sk[x_List] := With[{s = sk /@ x},
  With[{u = DeleteDuplicates[s]},
    Which[
      Length[x] <= 3 || Length[u] > 4 && Length[x] <= 12, s,
      Length[u] <= 16, Rep[Length[x], u],
      True, Rep[Length[x], Append[Take[u, 16], "..."]]]]];
sk[x_Association] := If[Length[x] > 12,
  Assoc[Length[x], DeleteDuplicates[sk /@ Keys[x]], DeleteDuplicates[sk /@ Values[x]]],
  Association[KeyValueMap[skKey[#1] -> sk[#2] &, x]]];
(* keys of small associations are column names: keep them *)
skKey[k_String] := k;
skKey[k_] := sk[k];
(* heads that stay inert with skeleton arguments are kept as they are *)
inert = Rule | RuleDelayed | Graphics | Graphics3D | GraphicsComplex | Polygon |
  Line | Point | Directive | EdgeForm | FaceForm | Raster | Missing;
sk[h_[args___]] := With[{s = sk /@ {args}},
  Which[
    MatchQ[h, inert], h @@ s,
    (* long argument sequences (a Plus of a thousand terms) compress like lists *)
    Length[s] > 12, Sk[h, sk[{args}]],
    True, Sk[h, Sequence @@ s]]];
sk[x_] := Head[x];

(* ---- summary statistics ---- *)
(* {min, max, mean, sd, distinct values, non-numeric entries, heads,
   decimal places} of the numbers in v *)
decimals[n_List] := SelectFirst[Range[0, 6],
  Function[k, AllTrue[n, Abs[# 10^k - Round[# 10^k]] < 10^-6 Max[1, Abs[# 10^k]] &]], 7];
numStats[v_List] := With[{n = Select[Flatten[v], NumericQ]},
  If[n === {}, None,
    Join[round /@ {Min[n], Max[n], Mean[n], If[Length[n] > 1, StandardDeviation[n], 0]},
      {Length[DeleteDuplicates[n]], Length[Flatten[v]] - Length[n], Union[Head /@ n],
       If[AllTrue[n, IntegerQ], 0, decimals[N[n]]]}]]];
(* Spearman rank correlations between the numeric columns of a table *)
rankCorr[rows_] := Module[{cols, num},
  cols = Transpose[rows];
  num = Select[Range[Length[cols]], AllTrue[cols[[#]], NumericQ] && Length[DeleteDuplicates[cols[[#]]]] > 1 &];
  If[Length[num] < 2 || Length[rows] < 3, None,
    {num, Round[SpearmanRho[N[Transpose[cols[[num]]]]], 0.01]}]];
(* statistics of one column: numeric summaries, or for a categorical column
   its labels and their frequencies — only when they are short labels (class
   names, factor levels); free text and images are summarized by skeleton *)
label[x_] := (StringQ[x] && StringLength[x] <= 40) || MatchQ[x, True | False | _Integer | _Missing];
oneCol[col_List] := Which[
  AllTrue[col, NumericQ[#] || MissingQ[#] &], {"N", numStats[col]},
  AllTrue[col, label] && Length[DeleteDuplicates[col]] <= 100,
    {"C", Length[DeleteDuplicates[col]], SortBy[Tally[col], -Last[#] &]},
  True, {"X", Length[DeleteDuplicates[col]], DeleteDuplicates[sk /@ col],
    If[AllTrue[col, StringQ], round /@ {Mean[StringLength[col]], Max[StringLength[col]]}, None]}];
colStats[rows_List] /; MatrixQ[rows, True &] := Table[oneCol[rows[[All, j]]], {j, Length[First[rows]]}];
colStats[v_List] /; VectorQ[v] := {oneCol[v]};
(* ragged rows: column j over the rows that have one *)
colStats[rows_List] /; AllTrue[rows, ListQ] := Table[
  oneCol[Cases[rows, r_ /; Length[r] >= j :> r[[j]]]], {j, Max[Length /@ rows]}];
colStats[_] := None;

imageStats[img_] := With[{d = ImageData[img, "Real"]},
  With[{px = If[ImageChannels[img] == 1, Flatten[d], Flatten[d, 1]]},
    If[ImageChannels[img] == 1, {round[Mean[px]], round[StandardDeviation[px]]},
      {round /@ Mean[px], round /@ StandardDeviation[px]}]]];
(* how well the image tiles: mean jump across the wrap-around seam relative to
   the mean jump between neighbouring pixels *)
tileStats[img_] := With[{d = ImageData[ColorConvert[img, "Grayscale"], "Real"]},
  round /@ {Mean[Abs[d[[1]] - d[[-1]]]] / Max[10^-6, Mean[Abs[d[[1]] - d[[2]]]]],
    Mean[Abs[d[[All, 1]] - d[[All, -1]]]] / Max[10^-6, Mean[Abs[d[[All, 1]] - d[[All, 2]]]]]}];

safe[expr_] := Quiet[Check[expr, $Failed]];
(* ExampleData[entry, prop], computed once per entry (some properties, like a
   large LP's "Equations", take minutes); the memo is dropped after each entry *)
ed[e_, p_] := ed[e, p] = safe[ExampleData[e, p]];
forget[] := (DownValues[ed] = {HoldPattern[ed[e_, p_]] :> (ed[e, p] = safe[ExampleData[e, p]])});
quiet[expr_] := Quiet[expr];
SetAttributes[quiet, HoldFirst];
SetAttributes[safe, HoldFirst];

(* ---- table assembly ---- *)
emit[coll_, columns_, rows_] := (
  WriteString[$out, StringRiffle[Prepend[columns, "@" <> coll], "\t"], "\n"];
  Scan[WriteString[$out, StringRiffle[#, "\t"], "\n"] &, rows]);

(* a column value: exact for metadata, a skeleton for data, empty when the
   entry does not have the property *)
cell[entry_, props_, p_, exact_] := If[!MemberQ[props, p], "",
  With[{v = ed[entry, p]},
    If[MemberQ[exact, p] || MatchQ[v, _Symbol | _Missing | _Integer | _Real],
      "=" <> fmt[v], fmt[sk[v]]]]];

(* the property whose value ExampleData[entry] returns *)
defaultProp[entry_, props_] := With[{d = safe[ExampleData[entry]]},
  SelectFirst[DeleteDuplicates[props], ed[entry, #] === d &, sk[d]]];

collect[coll_, names_, exact_, extra_] := Module[{entries, props, cols, rows, listed},
  entries = If[StringQ[#], {coll, #}, #] & /@ names;
  props = DeleteDuplicates[Flatten[safe[ExampleData[#, "Properties"]] & /@ entries]];
  listed = ExampleData[coll];
  cols = Join[{"@Properties", "@Default", "@Listed"}, props, Keys[extra]];
  rows = Table[
    With[{ps = safe[ExampleData[e, "Properties"]]},
      Join[{e[[2]], fmt[ps], fmt[defaultProp[e, ps]], fmt[MemberQ[listed, e]]},
        cell[e, ps, #, exact] & /@ props,
        fmt[quiet[safe[#[e]]]] & /@ Values[extra]] // (forget[]; #) &],
    {e, entries}];
  ClearSystemCache[];
  {cols, rows}];

run[coll_, names_, exact_, extra_] := With[{t = collect[coll, names, exact, extra]},
  emit[coll, t[[1]], t[[2]]]];

catalogue[coll_] := Last /@ ExampleData[coll];

(* ---- per-collection specs ---- *)
imageExtra = <|
  "@Options" -> (DeleteCases[Options[ExampleData[#]], MetaInformation -> _] &),
  "@GraphicsOptions" -> (With[{g = ExampleData[#, "Graphics"]},
      If[Head[g] === Graphics, {Rest[List @@ g[[1]]], Rest[List @@ g]}, Missing[]]] &),
  "@Stats" -> (imageStats[ExampleData[#]] &),
  "@Tile" -> (tileStats[ExampleData[#]] &)|>;

(* retired entries: no longer in the catalogue, still served with an
   ExampleData::obsolete warning *)
obsolete["TestImage"] = {"Lena", "TestPattern"};
spec["TestImage"] := {Join[catalogue["TestImage"], obsolete["TestImage"]],
  {"BitDepth", "ColorSpace", "DataType", "ImageSize", "Name"},
  Append[imageExtra, "@Obsolete" -> (Quiet[Check[ExampleData[#]; False, True,
    ExampleData::obsolete], ExampleData::obsolete] &)]};
spec["AerialImage"] := {catalogue["AerialImage"],
  {"BitDepth", "ColorSpace", "DataType", "ImageSize", "Name"}, imageExtra};
spec["Texture"] := {catalogue["Texture"],
  {"BitDepth", "ColorSpace", "DataType", "ImageSize", "Name"}, imageExtra};
spec["ColorTexture"] := {catalogue["ColorTexture"],
  {"BitDepth", "ColorSpace", "DataType", "ImageSize", "Name"}, imageExtra};

(* frame sub-entries are not in the catalogue but can be requested *)
withFrames[coll_] := Join[catalogue[coll],
  Flatten[Last /@ safe[ExampleData[{coll, #}, "Frames"]] & /@ catalogue[coll]]];
spec["TestImageSet"] := {withFrames["TestImageSet"],
  {"ColorSpace", "Frames", "ImageSize", "BitDepth", "ImageResolution", "StandardName", "Name"},
  <|"@Stats" -> (If[ImageQ[ExampleData[#]], imageStats[ExampleData[#]], None] &)|>};
spec["TestAnimation"] := {withFrames["TestAnimation"],
  {"ColorSpace", "Frames", "ImageSize", "BitDepth", "ImageResolution", "StandardName", "Name"},
  <|"@Stats" -> (If[ImageQ[ExampleData[#, "Image"]], imageStats[ExampleData[#, "Image"]], None] &),
    "@AnimationForm" -> (With[{a = ExampleData[#, "Animation"]},
      If[Head[a] === Manipulate, sk[a /. Graphics[r_Raster, o___] :> Graphics[Raster, o]], None]] &)|>};
spec["TestImage3D"] := {catalogue["TestImage3D"], {"DataType", "ImageSize", "Name"},
  <|"@Stats" -> (With[{d = Flatten[ImageData[ExampleData[#], "Real"]]},
      round /@ {Mean[d], StandardDeviation[d]}] &),
    "@Options" -> (DeleteCases[Options[ExampleData[#]], MetaInformation -> _] &)|>};

audioExtra = <|
  "@Stats" -> (With[{d = Flatten[AudioData[ExampleData[#, "Audio"]]]},
     round /@ {Sqrt[Mean[d^2]], Max[Abs[d]], Mean[d]}] &),
  "@AudioOptions" -> (Options[ExampleData[#, "Audio"]] &)|>;
spec["Audio"] := {catalogue["Audio"],
  {"Channels", "Duration", "Name", "SampleRate"}, audioExtra};
spec["Sound"] := {catalogue["Sound"],
  {"Channels", "Duration", "Name", "SampleRate"}, audioExtra};

spec["Geometry3D"] := {catalogue["Geometry3D"],
  {"BoundingBox", "ConnectedComponents", "Name", "PolygonCount"},
  <|"@FaceSizes" -> (Tally[Length /@ ExampleData[#, "PolygonData"]] &),
    "@Graphics3D" -> (With[{g = ExampleData[#, "Graphics3D"]},
      sk[g /. GraphicsComplex[v_, rest___] :> GraphicsComplex[Length[v], rest]]] &),
    "@Graphics3DOptions" -> (Options[ExampleData[#, "Graphics3D"]] &)|>};

spec["Dataset"] := {catalogue["Dataset"], {"Schema"},
  <|"@Keys" -> (With[{n = Normal[ExampleData[#]]},
      Which[AssociationQ[n], {Keys[n], Union @@ (Keys /@ Select[Values[n], AssociationQ])},
        ListQ[n] && AssociationQ[First[n]], {Length[n], Keys[First[n]]}, True, None]] &),
    "@Stats" -> (With[{n = Normal[ExampleData[#]]},
      If[ListQ[n] && AllTrue[n, AssociationQ], colStats[Values /@ n], None]] &)|>};

spec["Text"] := {catalogue["Text"], {"Author", "FullTitle", "Language", "Title"},
  <|"@Counts" -> (With[{s = ExampleData[#, "String"], l = ExampleData[#, "Lines"],
       f = ExampleData[#, "FormattedText"]},
      {StringLength[s], Length[ExampleData[#, "Words"]], Length[l],
       Count[l, ""], Length[TextSentences[s]], StringLength[f], StringCount[f, "\n\n"]}] &),
    "@Alphabet" -> (With[{c = Tally[Characters[ExampleData[#, "String"]]]},
      If[Length[c] > 200, {Length[c], MinMax[ToCharacterCode[c[[All, 1]]]]},
        StringJoin[SortBy[c, -Last[#] &][[All, 1]]]]] &),
    "@Cells" -> (With[{nb = ExampleData[#, "NotebookExpression"]},
      Tally[Cases[nb, Cell[_, style_String, ___] :> style, Infinity]]] &)|>};

spec["Statistics"] := {catalogue["Statistics"],
  {"ApplicationAreas", "ColumnHeadings", "ColumnTypes", "DataType", "Dimensions",
   "Name", "ObservationCount"},
  <|"@Stats" -> (colStats[ExampleData[#]] &),
    "@Corr" -> (With[{d = ExampleData[#]}, If[MatrixQ[d, True &], rankCorr[d], None]] &),
    "@Times" -> (With[{ts = ExampleData[#, "TimeSeries"]},
      If[Head[ts] === TemporalData, ts[[2, 2]] /. l_List /; Length[l] > 12 :> sk[l], None]] &),
    "@EventForm" -> (With[{ev = ExampleData[#, "EventData"]},
      If[MissingQ[ev], None, sk[ev]]] &)|>};

spec["MachineLearning"] := {catalogue["MachineLearning"],
  {"Dimensions", "LearningTask", "MissingData", "Name", "VariableTypes"},
  <|"@Stats" -> (With[{d = ExampleData[#, "Data"]},
      {colStats[Replace[d[[All, 1]], {a_Association :> Values[a], x_ /; !ListQ[x] :> {x}}, {1}]],
       colStats[d[[All, 2]]]}] &),
    "@Example" -> (sk[First[ExampleData[#, "Data"]]] &),
    "@Sizes" -> ({Length[ExampleData[#, "TrainingData"]], Length[ExampleData[#, "TestData"]]} &)|>};

matrixProps = {"AverageEntriesPerColumn", "AverageEntriesPerRow", "Bandwidth", "Collection",
  "Dimensions", "Entries", "Format", "ID", "LowerBandwidth", "MatrixStructure", "Name",
  "PatternSymmetry", "PositiveDefiniteQ", "StrongComponents", "StructuralFullRankQ",
  "StructuralRank", "Symmetry", "Type", "UpperBandwidth", "URL"};
(* Matrix data is not downloaded: its skeleton follows from the index *)
runMatrix[] := Module[{entries = ExampleData["Matrix"], props},
  props = DeleteDuplicates[Flatten[ExampleData[#, "Properties"] & /@ entries]];
  emit["Matrix", Join[{"@Properties"}, Complement[props, {"Matrix", "Source"}]],
    Table[Join[{e[[2]], fmt[ExampleData[e, "Properties"]]},
      "=" <> fmt[safe[ExampleData[e, #]]] & /@ Complement[props, {"Matrix", "Source"}]], {e, entries}]]];

lpExtra[kind_] := <|
  "@Form" -> (sk[ed[#, kind <> "Data"]] &),
  "@Nonzeros" -> (Length[ed[#, "ConstraintMatrix"]["NonzeroValues"]] &),
  "@Stats" -> (With[{d = ed[#, kind <> "Data"]},
     {numStats[d[[1]]], numStats[ed[#, "ConstraintMatrix"]["NonzeroValues"]]}] &),
  "@Variables" -> (With[{v = Variables[First[ed[#, "Equations"]]]},
     {Length[v], Take[v, UpTo[3]]}] &)|>;
spec["LinearProgramming"] := {catalogue["LinearProgramming"], {"Collection", "Dimensions", "Name"},
  lpExtra["LinearProgramming"]};
spec["LinearOptimization"] := {catalogue["LinearOptimization"], {"Collection", "Dimensions", "Name"},
  lpExtra["LinearOptimization"]};

spec["NetworkGraph"] := {catalogue["NetworkGraph"],
  {"ByteCount", "EdgeCount", "EdgeProperty", "Name", "StandardName", "VertexCount", "VertexProperty"},
  <|"@Graph" -> (With[{g = ExampleData[#, "FullGraph"]},
      {DirectedGraphQ[g], WeightedGraphQ[g], MultigraphQ[g], LoopFreeGraphQ[g],
       Length[ConnectedComponents[g]], round /@ {Min[#], Max[#], Mean[#]} &[VertexDegree[g]],
       Union[Head /@ VertexList[g]],
       If[WeightedGraphQ[g], numStats[N[AnnotationValue[g, EdgeWeight]]], None]}] &),
    "@GraphOptions" -> (With[{g = ExampleData[#, "Graph"]},
      sk[Options[g]]] &)|>};

(* ---- main ---- *)
$out = OpenWrite[$ScriptCommandLine[[2]], CharacterEncoding -> "UTF-8"];
colls = If[Length[$ScriptCommandLine] > 2, Drop[$ScriptCommandLine, 2], ExampleData[]];
WriteString[$out, "@@Collections\t", fmt[ExampleData[]], "\n"];
Do[
  Which[
    c === "Matrix", runMatrix[],
    c === "SpatialPointData",
      WriteString[$out, "@SpatialPointData\t@Catalogue\n", "\t", fmt[ExampleData[c]], "\n"],
    True, With[{s = spec[c]}, run[c, s[[1]], s[[2]], s[[3]]]]],
  {c, colls}];
Close[$out];
