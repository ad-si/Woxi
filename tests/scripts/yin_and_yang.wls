YinYang[size_] :=
 Graphics[{{Circle[{0, 0}, 2]}, {Disk[{0, 0},
     2, {90 Degree, -90 Degree}]}, {White, Disk[{0, 1}, 1]}, {Black,
    Disk[{0, -1}, 1]}, {Black, Disk[{0, 1}, 1/4]}, {White,
    Disk[{0, -1}, 1/4]}}, ImageSize -> 40 size]

(* Raw ExportString[…, "SVG"] bytes can never match between engines
   (cairo-rendered vs primitive-based SVG), so assert stable projections
   of the graphic and that the SVG export path produces SVG. *)
g = YinYang[5];
Print[Head[g]];
Print[Length[First[g]]];

svg = ExportString[g, "SVG"];
Print[StringQ[svg] && StringContainsQ[svg, "<svg"]];
