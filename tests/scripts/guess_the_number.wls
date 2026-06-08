(* The secret number would normally come from RandomInteger[{1, 10}] and the
   guesses from Input[]; both are hardcoded here. *)
number = 7;
guesses = {3, 9, 7};
Do[
  Print["Guess: ", g];
  If[g === number, Print["Well guessed!"]; Break[]],
  {g, guesses}]
