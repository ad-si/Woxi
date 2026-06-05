(* `x := RandomInteger[10]` normally yields a fresh random integer on each use.
   Hardcoded to a fixed value here so the snapshot is deterministic. *)
x := 7;
Print["<table>", "\n", "<tr><th></th><th>X</th><th>Y</th><th>Z</th></tr>"]
Scan[Print["<tr><td>", #, "</td><td>", x, "</td><td>", x, "</td><td>",
    "</td></tr>"] &, Range[3]]
Print["</table>"]
