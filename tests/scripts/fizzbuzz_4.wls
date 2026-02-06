SetAttributes[f, Listable]
f[n_ /; Mod[n, 15] == 0] := "FizzBuzz";
f[n_ /; Mod[n, 3] == 0] := "Fizz";
f[n_ /; Mod[n, 5] == 0] := "Buzz";
f[n_] := n;

f[Range[100]] // Map[Print]
