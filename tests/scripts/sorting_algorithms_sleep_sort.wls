(* Sleep sort starts one timer per element (delay proportional to value) so they
   print in sorted order; `RunScheduledTask` makes the timing asynchronous.
   Hardcoded to a direct sorted print here so the snapshot is deterministic. *)
Scan[Print[#] &, Sort[{1, 9, 8, 7, 6, 5, 3, 4, 5, 2, 0}]]
