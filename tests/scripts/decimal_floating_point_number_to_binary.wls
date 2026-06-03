dec2bin[x_] := Block[{digits, pos},
  {digits,  pos} = (RealDigits[ToExpression@x,  2] /. {a_, b_} :> {ToString /@ a, b});
  StringJoin @@ 
   Join@{Take[digits, pos], {"."}, Quiet@NestWhile[Most, Drop[digits, pos], Last@# === "0" &]}]

bin2dec[x_] := FromDigits[RealDigits@ToExpression@x, 2] // N

Print[NumberForm[#, {9, 5}], " => ", dec2bin@#] &@23.34375;
Print[NumberForm[#, {9, 5}], " => ", NumberForm[bin2dec@#, {9, 5}]] &@1011.11101;
