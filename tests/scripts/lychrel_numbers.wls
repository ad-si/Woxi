palindromeQ[n_Integer] := 
 Block[{digits = IntegerDigits[n]}, digits == Reverse@digits]

nextNumber[n_Integer] := n + FromDigits[Reverse@IntegerDigits@n]

lychrelQ[n_Integer] := ! 
  palindromeQ@
   Catch[Nest[If[palindromeQ[#], Throw[#], nextNumber[#]] &, 
     nextNumber[n], 500]]


