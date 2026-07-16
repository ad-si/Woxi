(*Ex1*)a = {{1, 3, 5}, {2, 4, 7}, {1, 1, 0}};
{l, u, p, c} = LUDecomposition[a];
Print[Normal[l]]
Print[Normal[u]]
Print[Normal[p]]
Print[c]
Print[Normal[p] . a == Normal[l] . Normal[u]]

(*Ex2*)a = {{11, 9, 24, 2}, {1, 5, 2, 6}, {3, 17, 18, 1}, {2, 5, 7, 1}};
{l, u, p, c} = LUDecomposition[a];
Print[Normal[l]]
Print[Normal[u]]
Print[Normal[p]]
Print[c]
Print[Normal[p] . a == Normal[l] . Normal[u]]
