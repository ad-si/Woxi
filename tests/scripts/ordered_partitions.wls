w[partitions_]:=Module[{s={},t=Total@partitions,list=partitions,k}, n=Length[list];
    While[n>0,s=Join[s,{Take[t,(k=First[list])]}];t=Drop[t,k];list=Rest[list];n--]; s]

m[p_]:=(Sort/@#)&/@(w[#,p]&/@Permutations[Range@Total[p]])//Union


(* ========== *)

Grid@m[{2, 0, 2}]

Grid@m[{1, 1, 1}]
