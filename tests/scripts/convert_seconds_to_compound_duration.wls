compoundDuration[x_Integer] :=
 StringJoin @@ (Riffle[
     ToString /@ ((({Floor[x/604800],
             Mod[x, 604800]} /. {a_, b_} -> {a, Floor[b/86400],
              Mod[b, 86400]}) /. {a__, b_} -> {a, Floor[b/3600],
            Mod[b, 3600]}) /. {a__, b_} -> {a, Floor[b/60],
          Mod[b, 60]}), {" wk, ", " d, ", " hr, ", " min, ",
      " sec"}] //. {a___, "0", b_, c___} -> {a, c})

Grid[Table[{n, "secs =",
   compoundDuration[n]}, {n, {7259, 86400, 6000000}}],
 Alignment -> {Left, Baseline}]
