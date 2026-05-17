rp[v_, pos_] := RotateRight[v, (Length[v] + 1)/2 - pos];
rho[m_] := MapIndexed[rp, m];
magic[n_] :=
  rho[Transpose[rho[Table[i*n + j, {i, 0, n - 1}, {j, 1, n}]]]];

square = magic[11] // Grid
Print["Magic number is ", Total[square[[1, 1]]]]
