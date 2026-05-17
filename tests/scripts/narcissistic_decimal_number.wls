narc[1] = 0;
narc[n_] :=
  narc[n] =
   NestWhile[# + 1 &, narc[n - 1] + 1,
    Plus @@ (IntegerDigits[#]^IntegerLength[#]) != # &];
narc /@ Range[25]
