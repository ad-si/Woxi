Horner[l_List, x_] := Fold[x #1 + #2 &, 0, l]
Print[Horner[{6, -4, 7, -19}, x]]
Print[Horner[{6, -4, 7, -19}, 3]]
