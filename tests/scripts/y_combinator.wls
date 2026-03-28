Y = Function[f, #[#] &[Function[g, f[g[g][##] &]]]];
factorial = Y[Function[f, If[# < 1, 1, # f[# - 1]] &]];
fibonacci = Y[Function[f, If[# < 2, #, f[# - 1] + f[# - 2]] &]];

Print[factorial[9]]
Print[fibonacci[9]]
