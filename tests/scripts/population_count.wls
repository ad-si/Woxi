popcount[n_Integer] := IntegerDigits[n, 2] // Total
Print["population count of powers of 3"]
popcount[#] & /@ (3^Range[0, 30])
(*******)
evilQ[n_Integer] := popcount[n] // EvenQ
evilcount = 0;
evillist = {};
i = 0;
While[evilcount < 30,
 If[evilQ[i], AppendTo[evillist, i]; evilcount++];
 i++
 ]
Print["first thirty evil numbers"]
evillist
(*******)
odiousQ[n_Integer] := popcount[n] // OddQ
odiouscount = 0;
odiouslist = {};
i = 0;
While[odiouscount < 30,
 If[odiousQ[i], AppendTo[odiouslist, i]; odiouscount++];
 i++
 ]
Print["first thirty odious numbers"]
odiouslist
