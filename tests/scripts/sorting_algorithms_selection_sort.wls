SelectSort[x_List] := Module[{n = 1, temp, xi = x, j},
  While[n <= Length@x,
   temp = xi[[n]];
   For[j = n, j <= Length@x, j++,
    If[xi[[j]] < temp, temp = xi[[j]]];
    ];
   xi[[n ;;]] = {temp}~Join~
     Delete[xi[[n ;;]], First@Position[xi[[n ;;]], temp] ];
   n++;
   ];
  xi
  ]


(* ========== *)

SelectSort2[x_List]:= Flatten[{Min@x, If[Length@x > 1, SelectSort2@Drop[x, First@Position[x, Min@x]], {}] }];


(* ========== *)

(* The lists and the number of trials would normally come from an RNG
   (`RandomInteger[150]` trials over `RandomInteger[150, RandomInteger[1000]]`
   element lists). They are fixed here because the draw decides the runtime:
   SelectSort's O(n^2) slice reassignment on a 1000-element list, 150 times
   over, runs for well past ten minutes. The values still span 0-150 and
   repeat the way a draw does. *)
{And @@ Table[l = Table[Mod[7 k^2 + 13 i, 151], {k, 120}];
   Through[And[Length@# == Length@SelectSort@# &, OrderedQ@SelectSort@# &]@l],
   {i, 4}],
 Block[{$RecursionLimit = Infinity},
  And @@ Table[l = Table[Mod[7 k^2 + 13 i, 151], {k, 120}];
    Through[And[Length@# == Length@SelectSort2@# &, OrderedQ@SelectSort2@# &]@l],
    {i, 4}]
  ]}
