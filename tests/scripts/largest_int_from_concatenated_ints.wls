makeLargestInt[list_] := Module[{sortedlist},
  sortedlist = Sort[list, Order[ToString[#1] <> ToString[#2], ToString[#2] <> ToString[#1]] < 0 &];
  Map[ToString, sortedlist] // StringJoin // FromDigits
  ]
(* testing with two examples *)
makeLargestInt[{1, 34, 3, 98, 9, 76, 45, 4}]
makeLargestInt[{54, 546, 548, 60}]
