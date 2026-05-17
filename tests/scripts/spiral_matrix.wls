AddSquareRing[x_List/;Equal@@Dimensions[x] && Length[Dimensions[x]]==2]:=Module[{new=x,size,smallest},
 size=Length[x];
 smallest=x[[1,1]];
 Do[
  new[[i]]=Prepend[new[[i]],smallest-i];
  new[[i]]=Append[new[[i]],smallest-3 size+i-3]
 ,{i,size}];
 PrependTo[new,Range[smallest-3size-3-size-1,smallest-3size-3]];
 AppendTo[new,Range[smallest-size-1,smallest-size-size-2,-1]];
 new
]
MakeSquareSpiral[size_Integer/;size>0]:=Module[{largest,start,times},
 start=size^2+If[Mod[size,2]==0,{{-4,-3},{-1,-2}},{{-1}}];
 times=If[Mod[size,2]==0,size/2-1,(size-1)/2];
 Nest[AddSquareRing,start,times]
]


(* ========== *)

MakeSquareSpiral[2] // MatrixForm
MakeSquareSpiral[7] // MatrixForm
