#!/usr/bin/env wolframscript
(* Generate the compact Natural Earth world-map blob used by GeoGraphics.

   Downloads the lowest-resolution (1:110m) Natural Earth admin-0 countries
   data set and converts it into a small binary file that is embedded into the
   Woxi binary via `include_bytes!`.

   Source: https://www.naturalearthdata.com  (public domain), distributed as
   GeoJSON by the nvkelso/natural-earth-vector mirror.

   Binary format (little-endian). Polygons are grouped by country so that
   individual countries can be looked up by name (for GeoNearest, country
   highlighting and choropleths):

       u32  num_countries
       repeat num_countries times:
           u16  name_len
           u8   name[name_len]      (UTF-8, the Natural Earth "NAME" field)
           u16  num_polygons
           repeat num_polygons times:
               u16  num_rings
               repeat num_rings times:
                   u16  num_points
                   repeat num_points times:
                       i16  lon * 100   (centi-degrees, range -18000..18000)
                       i16  lat * 100   (centi-degrees, range  -9000.. 9000)

   Run:  wolframscript -file scripts/gen_geo_data.wls
*)

url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/\
master/geojson/ne_110m_admin_0_countries.geojson";

(* Repo root is two levels up from this script (scripts/ -> repo). *)
repoRoot = DirectoryName[DirectoryName[$InputFileName]];
out = FileNameJoin[{repoRoot, "resources", "ne_110m_countries.bin"}];

Print["Downloading ", url];
data = Import[url, "RawJSON"];
features = data["features"];

(* Each polygon is a list of rings; each ring a list of {lon, lat} pairs. *)
polygonsOf[geom_] := Module[{type = geom["type"], c = geom["coordinates"]},
  Which[
    type === "Polygon", {c},
    type === "MultiPolygon", c,
    True, {}
  ]
];

(* Quantize a degree value to centi-degrees, clamped to the i16 range. *)
quant[x_] := Clip[Round[x*100], {-32768, 32767}];

stream = OpenWrite[out, BinaryFormat -> True];
BinaryWrite[stream, Length[features], "UnsignedInteger32", ByteOrdering -> -1];

npoints = 0; npolys = 0;
Do[
  Module[{name, nameBytes, polys},
    name = feature["properties"]["NAME"];
    If[!StringQ[name], name = ""];
    nameBytes = ToCharacterCode[name, "UTF8"];
    BinaryWrite[stream, Length[nameBytes], "UnsignedInteger16", ByteOrdering -> -1];
    BinaryWrite[stream, nameBytes, "UnsignedInteger8"];
    polys = polygonsOf[feature["geometry"]];
    npolys += Length[polys];
    BinaryWrite[stream, Length[polys], "UnsignedInteger16", ByteOrdering -> -1];
    Do[
      BinaryWrite[stream, Length[poly], "UnsignedInteger16", ByteOrdering -> -1];
      Do[
        BinaryWrite[stream, Length[ring], "UnsignedInteger16", ByteOrdering -> -1];
        (* lon, lat interleaved as one Integer16 list — one write per ring. *)
        coords = Flatten[{quant[#[[1]]], quant[#[[2]]]} & /@ ring];
        BinaryWrite[stream, coords, "Integer16", ByteOrdering -> -1];
        npoints += Length[ring];
        ,
        {ring, poly}
      ];
      ,
      {poly, polys}
    ];
  ];
  ,
  {feature, features}
];
Close[stream];

Print["Wrote ", out, " (", FileByteCount[out], " bytes, ",
  Length[features], " countries, ", npolys, " polygons, ", npoints, " points)"];
