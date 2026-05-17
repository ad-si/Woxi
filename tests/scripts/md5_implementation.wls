md5[string_String] :=
 Module[{r = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17,
     22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4,
     11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
     15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21},
   k = Table[Floor[2^32*Abs@Sin@i], {i, 1, 64}], h0 = 16^^67452301,
   h1 = 16^^efcdab89, h2 = 16^^98badcfe, h3 = 16^^10325476,
   data = Partition[
     Join[FromDigits[Reverse@#, 256] & /@
         Partition[
          PadRight[Append[#, 128], Mod[56, 64, Length@# + 1]], 4],
        Reverse@IntegerDigits[8 Length@#, 2^32, 2]] &@
      ImportString[string, "Binary"], 16], a, b, c, d, f, g},
  Do[{a, b, c, d} = {h0, h1, h2, h3};
   Do[Which[1 <= i <= 16,
     f = BitOr[BitAnd[b, c], BitAnd[BitNot[b], d]]; g = i - 1,
     17 <= i <= 32, f = BitOr[BitAnd[d, b], BitAnd[BitNot[d], c]];
     g = Mod[5 i - 4, 16], 33 <= i <= 48, f = BitXor[b, c, d];
     g = Mod[3 i + 2, 16], 49 <= i <= 64,
     f = BitXor[c, BitOr[b, BitNot[d] + 2^32]];
     g = Mod[7 i - 7, 16]]; {a, b, c, d} = {d,
      BitOr[BitShiftLeft[#1, #2], BitShiftRight[#1, 32 - #2]] &[
        Mod[a + f + k[[i]] + w[[g + 1]], 2^32], r[[i]]] + b, b,
      c}, {i, 1, 64}]; {h0, h1, h2, h3} =
    Mod[{h0, h1, h2, h3} + {a, b, c, d}, 2^32], {w, data}];
  "0x" ~~ IntegerString[
    FromDigits[
     Flatten[Reverse@IntegerDigits[#, 256, 4] & /@ {h0, h1, h2, h3}],
     256], 16, 32]]


