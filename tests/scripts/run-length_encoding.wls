RunLengthEncode[input_String]:=StringJoin@@Sequence@@@({ToString @Length[#],First[#]}&/@Split[Characters[input]])
RunLengthDecode[input_String]:=StringJoin@@ConstantArray@@@Reverse/@Partition[(Characters[input]/.(ToString[#]->#&/@Range[0,9]))//.{x___,i_Integer,j_Integer,y___}:>{x,10i+j,y},2]


(* ========== *)

mystring="WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWBWWWWWWWWWWWWWW";
RunLengthEncode[mystring]
RunLengthDecode[%]
%==mystring


(* ========== *)

12W1B12W3B24W1B14W
WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWBWWWWWWWWWWWWWW
True


(* ========== *)

RunLengthEncode[s_String] := StringJoin[
  {ToString[Length[#]] <> First[#]} & /@ Split[StringSplit[s, ""]]
  ]

RunLengthDecode[s_String] := StringJoin[
  Table[#[[2]], {ToExpression[#[[1]]]}] & /@
   Partition[StringSplit[s, x : _?LetterQ :> x], 2]
  ]
