PerfectQ[i_Integer] := Total[Divisors[i]] == 2 i


(* ========== *)

PerfectQ[496]
PerfectQ[128]
Flatten[PerfectQ/@Range[10000]//Position[#,True]&]


(* ========== *)

True
False
{6,28,496,8128}
