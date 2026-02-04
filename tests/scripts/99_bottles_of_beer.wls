Bottle[n_] := ToString[n] <> If[n==1," bottle"," bottles"] <> " of beer"

BottleSong[n_] := Print[
  Bottle[n] <> " on the wall," <>
  Bottle[n] <>
  ", take one down, pass it around, " <>
  Bottle[n-1] <> " on the wall."
]

BottleSong /@ Range[99,1,-1]
