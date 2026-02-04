Do[
  Print[
    Which[
      Mod[i, 15] == 0, "FizzBuzz",
      Mod[i, 5] == 0, "Buzz",
      Mod[i, 3] == 0, "Fizz",
      True, i
    ]
  ],
  {i, 100}
]
