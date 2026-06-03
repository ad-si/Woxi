(* defining functions *)
setcc[a_, b_] := a <= x <= b
setoo[a_, b_] := a < x < b
setco[a_, b_] := a <= x < b
setoc[a_, b_] := a < x <= b
setSubtract[s1_, s2_] := s1  &&  Not[s2];  (* new function; subtraction not built in *)
inSetQ[y_, set_] := set /. x -> y
(* testing sets *)
set1 = setoc[0, 1]  || setco[0, 2]  (* union built in as || shortcut (OR) *);
Print[set1]
Print["First trial set, (0, 1] \[Union] [0, 2) , testing for {0,1,2}:"]
Print[inSetQ[#, set1] & /@ {0, 1, 2}]
set2 = setco[0, 2] && setoc[1, 2];  (* intersection built in as && shortcut (AND) *)
Print[]
Print[set2]
Print["Second trial set, [0, 2) \[Intersection] (1, 2], testing for {0,1,2}:"]
Print[inSetQ[#, set2] & /@ {0, 1, 2}]
Print[]
set3 = setSubtract[setco[0, 3], setoo[0, 1]];
Print[set3]
Print["Third trial set, [0, 3) \[Minus] (0, 1), testing for {0,1,2}"]
Print[inSetQ[#, set3] & /@ {0, 1, 2}]
Print[]
set4 = setSubtract[setco[0, 3], setcc[0, 1]];
Print[set4]
Print["Fourth trial set, [0,3)\[Minus][0,1], testing for {0,1,2}:"]
Print[inSetQ[#, set4] & /@ {0, 1, 2}]
