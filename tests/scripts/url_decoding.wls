URLDecoding[url_] :=
 StringReplace[url, "%" ~~ x_ ~~ y_ :> FromDigits[x ~~ y, 16]] //.
  StringExpression[x___, Longest[n__Integer], y___] :>
   StringExpression[x, FromCharacterCode[{n}, "UTF8"], y]


