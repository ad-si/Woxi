RandomSample[{1, 2, 3, 4, 5, 6}]


(* ========== *)

Shuffle[input_List /; Length[input] >= 1] :=
 Module[{indices = {}, allindices = Range[Length[input]]},
  Do[
   AppendTo[indices,
     Complement[allindices, indices][[RandomInteger[{1, i}]]]];
   ,
   {i, Length[input], 1, -1}
   ];
  input[[indices]]
  ]


(* ========== *)

Shuffle[{1, 2, 3, 4, 5, 6}]
