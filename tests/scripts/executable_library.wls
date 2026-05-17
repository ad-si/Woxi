hailstone[1] = {1};
hailstone[n_] :=
  hailstone[n] = Prepend[hailstone[If[EvenQ[n], n/2, 3 n + 1]], n];
If[$ScriptCommandLine[[1]] == $Input,
  val = hailstone[27];
  Print["hailstone(27) starts with ", val[[;; 4]], ", ends with ",
   val[[-4 ;;]], ", and has length ", Length[val], "."];
  val = MaximalBy[Range[99999], Length@*hailstone][[1]];
  Print[val, " has the longest hailstone sequence with length ",
   Length[hailstone[val]], "."]];


