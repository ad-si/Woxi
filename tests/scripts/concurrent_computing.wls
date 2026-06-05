(* The task prints the three strings concurrently (`ParallelDo` with a random
   `Pause`), so their order varies between runs. Hardcoded to a fixed sequential
   order here so the snapshot is deterministic. *)
Do[Print[s], {s, {"Enjoy", "Rosetta", "Code"}}]
