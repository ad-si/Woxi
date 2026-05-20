x=1+2I;
y=3+4I;
Print[x+y];
Print[x-y];
Print[y x];
Print[y/x];
Print[x^3];
Print[y^4];
Print[x^y];
(* Round to 10 decimals so the snapshot is stable across libm
   implementations — Linux glibc and macOS libm differ in the last
   ULP for transcendentals like exp(...)*cos(...). *)
r = N[x^y];
Print[N[Round[Re[r], 10^-10] + I Round[Im[r], 10^-10]]];
