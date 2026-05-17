Hofstadter[1] = Hofstadter[2] = 1;
Hofstadter[n_Integer?Positive] := Hofstadter[n] = Block[{$RecursionLimit = Infinity},
   Hofstadter[n - Hofstadter[n - 1]] + Hofstadter[n - Hofstadter[n - 2]]
]


