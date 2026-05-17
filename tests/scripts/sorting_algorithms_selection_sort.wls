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

{And @@ Table[l = RandomInteger[150, RandomInteger[1000]];
   Through[And[Length@# == Length@SelectSort@# &, OrderedQ@SelectSort@# &]@l],
   {RandomInteger[150]}],
 Block[{$RecursionLimit = Infinity},
  And @@ Table[l = RandomInteger[150, RandomInteger[1000]];
    Through[And[Length@# == Length@SelectSort2@# &, OrderedQ@SelectSort2@# &]@l],
    {RandomInteger[150]}]
  ]}
