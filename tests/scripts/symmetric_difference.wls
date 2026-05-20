mySymDiff[x_List, y_List] := Join[Complement[x, Intersection[x, y]], Complement[y, Intersection[x, y]]]
Print[mySymDiff[{1, 2, 3, 4}, {3, 4, 5, 6}]]
Print[mySymDiff[{1, 2, 3, 4, 5}, {3, 5, 7}]]
