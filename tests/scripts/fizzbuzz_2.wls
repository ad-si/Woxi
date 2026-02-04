fizz[i_] := Mod[i, 3] == 0
buzz[i_] := Mod[i, 5] == 0

Range[100] /. {
  i_ /; fizz[i] && buzz[i] :> "FizzBuzz",
  i_?fizz :> "Fizz",
  i_?buzz :> "Buzz"
} // Map[Print]
