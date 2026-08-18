(* ::Package:: *)

(* Sync tests/SUMMARY.md and the `nav` block of tests/zensical.toml.

   Two responsibilities:

   1. Augment SUMMARY.md so every per-function page is listed under its
      category. For any entry whose target file has a sibling directory
      of per-function pages (e.g. `math/bits.md` <-> `math/bits/`), the
      function pages are inserted as nested children below the category
      entry. Files already referenced are skipped so each page appears
      exactly once. Entries pointing at a `.md` file that no longer
      exists are dropped — otherwise a moved page leaves a nav link that
      404s on the published site and a dead `Information[…]` URL.
      Idempotent — running it again after function pages are
      added/moved/removed updates the tree.

   2. Emit a TOML `nav` array into tests/zensical.toml, replacing the
      lines between the `# nav-begin` and `# nav-end` markers. The nav
      mirrors the SUMMARY structure exactly. *)

scriptDir = DirectoryName[$InputFileName];
root = DirectoryName[scriptDir];
testsDir = FileNameJoin[{root, "tests"}];
docsDir = FileNameJoin[{testsDir, "cli"}];
summaryFile = FileNameJoin[{testsDir, "SUMMARY.md"}];
zensicalFile = FileNameJoin[{testsDir, "zensical.toml"}];

navBegin = "# nav-begin";
navEnd = "# nav-end";

(* ----- IO helpers -------------------------------------------------- *)

(* Read/write through raw bytes with explicit UTF-8 conversion.
   ReadString/WriteString in wolframscript ignore the CharacterEncoding
   option and interpret bytes as Latin-1, mangling non-ASCII content. *)
readText[file_String] := Module[{bytes},
  bytes = BinaryReadList[file, "Byte"];
  If[bytes === {} || bytes === EndOfFile, "",
    FromCharacterCode[bytes, "UTF8"]]];

writeText[file_String, content_String] := Module[{strm},
  strm = OpenWrite[file, BinaryFormat -> True];
  BinaryWrite[strm, ToCharacterCode[content, "UTF8"]];
  Close[strm];];

writeTextIfChanged[file_String, newText_String, text_String] :=
  If[newText =!= text,
    writeText[file, newText]; Print["wrote ", file],
    Print[file, " unchanged"]];

(* Split text into lines, preserving empty lines, dropping a single
   trailing empty line if the text ends with "\n" — same semantics as
   Python's str.splitlines(). *)
splitLines[s_String] := Module[{parts, current},
  parts = {};
  current = "";
  Do[
    If[c === "\n",
      AppendTo[parts, current]; current = "",
      current = current <> c],
    {c, Characters[s]}];
  If[current =!= "" || !StringEndsQ[s, "\n"], AppendTo[parts, current]];
  parts];

(* Byte-wise lexicographic less-than on strings, matching Python's
   sorted() over ASCII paths. Wolfram's default Sort uses canonical
   order (length-first on character-code lists, case-folded on strings),
   neither of which matches Python. *)
asciiLess[s1_String, s2_String] := Catch[Module[
    {a = ToCharacterCode[s1], b = ToCharacterCode[s2], n},
    n = Min[Length[a], Length[b]];
    Do[
      Which[
        a[[i]] < b[[i]], Throw[True],
        a[[i]] > b[[i]], Throw[False]],
      {i, n}];
    Length[a] < Length[b]]];

(* ----- Outline parsing -------------------------------------------- *)

(* parseEntry[line] returns {indent, label, target} or None. *)
parseEntry[line_String] := Module[{r},
  r = StringCases[line,
    StartOfString ~~ ind:((" " | "\t")...) ~~ "- [" ~~
      lbl:(Except["]"]..) ~~ "](" ~~ tgt:(Except[")"]..) ~~ ")" ~~
      ((" " | "\t")...) ~~ EndOfString :> {ind, lbl, tgt}, 1];
  If[r === {}, None, First[r]]];

(* ----- Summary augmentation --------------------------------------- *)

(* A `.md` target that has no file behind it. Non-`.md` targets (e.g.
   the hand-written `jupyterlite-showcase.html` page) are left alone. *)
danglingQ[target_String] := Module[{targetPath},
  If[!StringEndsQ[target, ".md"], Return[False]];
  targetPath = FileNameJoin[{docsDir, target}];
  !FileExistsQ[targetPath] || FileType[targetPath] =!= File];

updateSummary[text_String] := Module[{
    lines, referenced, out, ent, indent, target,
    targetPath, siblingDir, children, childIndent, rel, newText
  },
  lines = splitLines[text];
  referenced = Association[];
  Do[
    With[{e = parseEntry[line]},
      If[e =!= None, referenced[e[[3]]] = True]],
    {line, lines}];
  out = {};
  Do[
    ent = parseEntry[line];
    If[ent =!= None && danglingQ[ent[[3]]],
      WriteString["stderr",
        "dropped dangling entry: " <> ent[[3]] <> "\n"];
      Continue[]];
    AppendTo[out, line];
    If[ent === None, Continue[]];
    indent = ent[[1]]; target = ent[[3]];
    If[!StringEndsQ[target, ".md"], Continue[]];
    targetPath = FileNameJoin[{docsDir, target}];
    siblingDir = StringDrop[targetPath, -3];
    If[!DirectoryQ[siblingDir], Continue[]];
    children = Sort[FileNames["*.md", siblingDir], asciiLess];
    childIndent = indent <> "  ";
    Do[
      rel = StringReplace[
        StringDrop[child, StringLength[docsDir] + 1], "\\" -> "/"];
      If[KeyExistsQ[referenced, rel], Continue[]];
      If[DirectoryQ[StringDrop[child, -3]], Continue[]];
      AppendTo[out,
        childIndent <> "- [`" <> FileBaseName[child] <> "`]("
          <> rel <> ")"],
      {child, children}],
    {line, lines}];
  newText = StringRiffle[out, "\n"] <> "\n";
  newText];

(* ----- Outline -> tree -------------------------------------------- *)

(* Tree node: <|"label" -> _, "target" -> _, "children" -> {...}|> *)
createNode[lbl_String, tgt_String, ch_List] :=
  <|"label" -> lbl, "target" -> tgt, "children" -> ch|>;

parseOutline[text_String] := Module[{entries},
  entries = DeleteCases[
    Map[
      With[{e = parseEntry[#]},
        If[e === None, Nothing,
          {StringLength[e[[1]]], e[[2]], e[[3]]}]] &,
      splitLines[text]],
    Nothing];
  parseLevel[entries, -1][[1]]];

(* parseLevel[entries, minDepth] consumes all leading entries with
   depth strictly greater than minDepth and returns
   {nodeList, leftoverEntries}. *)
parseLevel[{}, _] := {{}, {}};
parseLevel[entries_List, minDepth_Integer] := Module[
  {first, rest, ch, after},
  first = First[entries];
  If[first[[1]] <= minDepth, Return[{{}, entries}]];
  {ch, after} = parseLevel[Rest[entries], first[[1]]];
  {rest, after} = parseLevel[after, minDepth];
  {Prepend[rest, createNode[first[[2]], first[[3]], ch]], after}];

(* ----- Tree -> TOML ----------------------------------------------- *)

tomlString[s_String] :=
  "\"" <> StringReplace[s,
    {"\\" -> "\\\\", "\"" -> "\\\""}] <> "\"";

(* Zensical does not render markdown in nav titles, so backticks would
   surface as literal characters in the sidebar. *)
stripBackticks[label_String] := StringReplace[label, "`" -> ""];

emitNavLines[tree_List, depth_Integer] := Module[
  {pad, lines, target, label, ch, lstr},
  pad = StringJoin[ConstantArray["  ", depth]];
  lines = {};
  Do[
    target = node["target"];
    label = stripBackticks[node["label"]];
    ch = node["children"];
    If[!StringEndsQ[target, ".md"],
      (* Drop entry; hoist children so their pages remain reachable. *)
      If[ch =!= {}, lines = Join[lines, emitNavLines[ch, depth]]];
      Continue[]];
    lstr = tomlString[label];
    If[ch =!= {},
      AppendTo[lines, pad <> "{ " <> lstr <> " = ["];
      AppendTo[lines, pad <> "  \"" <> target <> "\","];
      lines = Join[lines, emitNavLines[ch, depth + 1]];
      AppendTo[lines, pad <> "] },"],
      AppendTo[lines,
        pad <> "{ " <> lstr <> " = \"" <> target <> "\" },"]],
    {node, tree}];
  lines];

renderNavBlock[tree_List] :=
  StringRiffle[
    Join[{"nav = ["}, emitNavLines[tree, 1], {"]"}], "\n"];

spliceNav[tomlText_String, navBlock_String] := Module[
  {lines, posBegin, posEnd, iBegin, iEnd, before, after, navLines, suffix},
  lines = splitLines[tomlText];
  posBegin = FirstPosition[lines,
    s_String /; StringTrim[s] === navBegin, Missing[]];
  If[MissingQ[posBegin],
    WriteString["stderr",
      zensicalFile <> ": missing `" <> navBegin
        <> "` / `" <> navEnd <> "` markers\n"];
    Exit[1]];
  iBegin = First[posBegin];
  posEnd = FirstPosition[Drop[lines, iBegin],
    s_String /; StringTrim[s] === navEnd, Missing[]];
  If[MissingQ[posEnd],
    WriteString["stderr",
      zensicalFile <> ": missing `" <> navEnd <> "` marker\n"];
    Exit[1]];
  iEnd = First[posEnd] + iBegin;
  before = Take[lines, iBegin];
  after = Drop[lines, iEnd - 1];
  navLines = splitLines[navBlock];
  suffix = If[StringEndsQ[tomlText, "\n"], "\n", ""];
  StringRiffle[Join[before, navLines, after], "\n"] <> suffix];

(* ----- Main ------------------------------------------------------- *)

summaryText = readText[summaryFile];
newSummary = updateSummary[summaryText];
writeTextIfChanged[summaryFile, newSummary, summaryText];

tree = parseOutline[newSummary];
navBlock = renderNavBlock[tree];

tomlText = readText[zensicalFile];
newToml = spliceNav[tomlText, navBlock];
writeTextIfChanged[zensicalFile, newToml, tomlText];
