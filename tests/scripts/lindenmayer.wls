lindenmayer[i_, b_, h_, j_, r_, n_] :=
 (a = h; p = j; s = k = {}; t = Flatten;
  (Switch[#,
      6, s = {a, p, s},
      8, {a, p, s} = s,
      _C, k = {k, Line@{p, p += {Cos@a, Sin@a}}}];
     If[# < 9, a += I^# b ]) & /@ t@Nest[# /. r &, i, n];
  Graphics@t@k);
lindenmayer[{C[1], X}, Pi/2, 0, {0, 0}, {X -> {X, 4, Y, C[1]}, Y -> {C[1], X, 2, Y}}, 10]
