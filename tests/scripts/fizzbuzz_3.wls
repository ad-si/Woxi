Table[
  If[# === "", i, #]& @ StringJoin[
    Table[
      If[Divisible[i, First@nw], Last@nw, ""],
      {nw, {{3, "Fizz"}, {5, "Buzz"}}}
    ]
  ],
  {i, 1, 100}
] // Map[Print]
