seq[n_] := 
  NestList[If[# == 0, 0, 
     DivisorSum[#, # &, Function[div, div != #]]] &, n, 16];
class[seq_] := 
  Which[Length[seq] < 2, "Non-terminating", MemberQ[seq, 0], 
   "Terminating", seq[[1]] == seq[[2]], "Perfect", 
   Length[seq] > 2 && seq[[1]] == seq[[3]], "Amicable", 
   Length[seq] > 3 && MemberQ[seq[[4 ;;]], seq[[1]]], "Sociable", 
   MatchQ[class[Rest[seq]], "Perfect" | "Aspiring"], "Aspiring", 
   MatchQ[class[Rest[seq]], "Amicable" | "Sociable" | "Cyclic"], 
   "Cyclic", True, "Non-terminating"];
notate[seq_] := 
  Which[seq == {}, {}, 
   MemberQ[Rest[seq], 
    seq[[1]]], {Prepend[TakeWhile[Rest[seq], # != seq[[1]] &], 
     seq[[1]]]}, True, Prepend[notate[Rest[seq]], seq[[1]]]];
Print[{#, class[seq[#]], notate[seq[#]] /. {0} -> 0}] & /@ {1, 2, 3, 4, 5, 6, 7, 
   8, 9, 10, 11, 12, 28, 496, 220, 1184, 12496, 1264460, 790, 909, 
   562, 1064, 1488, 15355717786080};
