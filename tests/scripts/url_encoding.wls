URLEncoding[url_] :=
 StringReplace[url,
  x : Except[
     Join[CharacterRange["0", "9"], CharacterRange["a", "z"],
      CharacterRange["A", "Z"]]] :>
   StringJoin[("%" ~~ #) & /@
     IntegerString[ToCharacterCode[x, "UTF8"], 16]]]


