Print[BSDrand[x_] := Mod[x*1103515245 + 12345, 2147483648]];
Print[NestList[BSDrand, 0, 10]];
Print[MSrand[x_] := Mod[x*214013 + 2531011, 2147483648]];
Print[BitShiftRight[ NestList[MSrand, 0, 10], 16]];
