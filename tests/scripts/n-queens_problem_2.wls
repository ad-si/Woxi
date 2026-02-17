n = 6;
cnt = 1;
per = Permutations[Range[n], {n}];  (* All permutations of length n *)

(* Riffle in the reverse of Range[n], partitioned into pairs *)
Do[
  per[[q]] = Partition[Riffle[Reverse[Range[n]], per[[q]]], 2],
  {q, 1, Length[per]}
];

Do[
  w = Subsets[per[[t]], {2}];  (* All pairs from the set of coordinate pairs *)
  tot = 0;
  (* If abs(x1-x2) == abs(y1-y2), the queens share a diagonal *)
  Do[
    y = Abs[w[[q, 1, 1]] - w[[q, 2, 1]]];
    x = Abs[w[[q, 1, 2]] - w[[q, 2, 2]]];
    If[x == y, tot++],
    {q, 1, Length[w]}
  ];
  (* If no diagonal clashes, print the solution *)
  If[tot == 0,
    g = Grid[Table[" ", {n}, {n}], Alignment -> Center, Frame -> All, Spacings -> {1.2, 1}];
    Do[g[[1, per[[t, w, 1]], per[[t, w, 2]]]] = "Q", {w, 1, n}];
    Print[cnt, "   ", per[[t]], "   ", g];
    cnt++
  ],
  {t, 1, Length[per]}
]
