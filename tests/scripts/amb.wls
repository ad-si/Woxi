CheckValid[i_List]:=If[Length[i]<=1,True,And@@(StringTake[#[[1]],-1]==StringTake[#[[2]],1]&/@Partition[i,2,1])]
 sets={{"the","that","a"},{"frog","elephant","thing"},{"walked","treaded","grows"},{"slowly","quickly"}};
 Select[Tuples[sets],CheckValid]


(* ========== *)

{{"that", "thing", "grows", "slowly"}}


(* ========== *)

CheckValid2[i_List] := StringFreeQ[StringJoin[Riffle[i, ","]], a_ ~~ "," ~~ b_ /; a =!= b]
