Which[
  Mod[#, 15] == 0, "FizzBuzz",
  Mod[#, 3] == 0, "Fizz",
  Mod[#, 5] == 0, "Buzz",
  True, #
]& /@ Range[1, 100] // Map[Print]
