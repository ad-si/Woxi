(*note, this usage of Module allows us to memoize FibonacciWord
  without exposing it to the global scope*)
Module[{FibonacciWord, step},
  FibonacciWord[1] = "1";
  FibonacciWord[2] = "0";
  FibonacciWord[n_Integer?(# > 2 &)] :=
   (FibonacciWord[n] = FibonacciWord[n - 1] <> FibonacciWord[n - 2]);

  step["0", {_?EvenQ}] = N@RotationTransform[Pi/2];
  step["0", {_?OddQ}] = N@RotationTransform[-Pi/2];
  step[___] = Identity;

  FibonacciFractal[n_] := Module[{steps, dirs},
    steps = MapIndexed[step, Characters[FibonacciWord[n]]];
    dirs = ComposeList[steps, {0, 1}];
    Graphics[Line[FoldList[Plus, {0, 0}, dirs]]]]];
