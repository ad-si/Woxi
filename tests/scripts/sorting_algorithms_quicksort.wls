QuickSort[x_List] := Module[{pivot},
  If[Length@x <= 1, Return[x]];
  pivot = RandomChoice@x;
  Flatten@{QuickSort[Cases[x, j_ /; j < pivot]], Cases[x, j_ /; j == pivot], QuickSort[Cases[x, j_ /; j > pivot]]}
  ]


(* ========== *)

qsort[{}] = {};
qsort[{x_, xs___}] := Join[qsort@Select[{xs}, # <= x &], {x}, qsort@Select[{xs}, # > x &]];


(* ========== *)

QuickSort[{}] := {}
QuickSort[list: {__}] := With[{pivot=RandomChoice[list]},
	Join[ <|1->{}, -1->{}|>, GroupBy[list,Order[#,pivot]&] ] // Catenate[ {QuickSort@#[1], #[0], QuickSort@#[-1]} ]&
]
