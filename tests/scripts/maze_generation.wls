MazeGraphics[m_, n_] :=
 Block[{$RecursionLimit = Infinity,
   unvisited = Tuples[Range /@ {m, n}], maze},
  maze = Graphics[{Line[{{#, # - {0, 1}}, {#, # - {1, 0}}}] & /@
      unvisited,
     Line[{{0, n}, {0, 0}, {m, 0}}]}]; {unvisited =
      DeleteCases[unvisited, #];
     Do[If[MemberQ[unvisited, neighbor],
       maze = DeleteCases[
         maze, {#,
           neighbor - {1, 1}} | {neighbor, # - {1, 1}}, {5}]; #0@
        neighbor], {neighbor,
       RandomSample@{# + {0, 1}, # - {0, 1}, # + {1, 0}, # - {1,
           0}}}]} &@RandomChoice@unvisited; maze];
maze = MazeGraphics[21, 13]


