shE[s_String] := -Plus @@ ((# Log[2., #]) & /@ ((Length /@ Gather[#])/
         Length[#]) &[Characters[s]])


(* ========== *)

shE["1223334444"]
1.84644
shE["Rosetta Code"]
3.08496
