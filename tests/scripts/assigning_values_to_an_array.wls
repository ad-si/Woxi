a={x,y,z};
 a[[2]]=7;
 a


(* ========== *)

{x, 7, z}


(* ========== *)

SaveAssign[list_, index_Integer, value_] := Module[{x = list},
  If[index > Length[list] || index < 0,
   Print["index out of range"];
   ,
   x[[index]] = value;
   ];
  x
  ]


(* ========== *)

a={x,y,z};
 SaveAssign[a,2,7]
 SaveAssign[a,4,7]


(* ========== *)

{x, 7, z}
index out of range
{x,y,z}
