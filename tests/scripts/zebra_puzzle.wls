ClearAll[EliminatePoss, FilterPuzzle]
EliminatePoss[ct_, key1_, key2_] := Module[{t = ct, poss1, poss2, poss, notposs},
  poss1 = Position[t, key1];
  poss2 = Position[t, key2];
  poss = Intersection[Last /@ poss1, Last /@ poss2];
  notposs = Complement[Range[5], poss];
  poss1 = Select[poss1, MemberQ[notposs, Last[#]] &];
  poss2 = Select[poss2, MemberQ[notposs, Last[#]] &];
  t = ReplacePart[t, poss1 -> Null];
  t = ReplacePart[t, poss2 -> Null];
  t
]
FilterPuzzle[tbl_] := Module[{t = tbl, poss1, poss2, poss, notposs, rows, columns, vals, sets, delpos},
  t = EliminatePoss[t, "English", "Red"]; (*2. The English man lives in the red house. *)
  t = EliminatePoss[t, "Swede", "Dog"]; (* 3. The Swede has a dog. *)
  t = EliminatePoss[t, "Dane", "Tea"]; (* 4. The Dane drinks tea. *)
  t = EliminatePoss[t, "Green", "Coffee"]; (* 6. They drink coffee in the green house. *)
  t = EliminatePoss[t, "Pall Mall", "Birds"]; (* 7. The man who smokes Pall Mall has birds.*)
  t = EliminatePoss[t, "Yellow", "Dunhill"]; (* 8. In the yellow house they smoke Dunhill. *)
  t = EliminatePoss[t, "Blue Master", "Beer"]; (*13. The man who smokes Blue Master drinks beer. *)
  t = EliminatePoss[t, "German", "Prince"]; (* 14. The German smokes Prince. *)

  (* 9. In the middle house they drink milk. *)
  poss = Position[t, "Milk"];
  delpos = Select[poss, #[[2]] != 3 &];
  t = ReplacePart[t, delpos -> Null];

  (* 10. The Norwegian lives in the first house. *)
  poss = Position[t, "Norwegian"];
  delpos = Select[poss, #[[2]] != 1 &];
  t = ReplacePart[t, delpos -> Null];

  (* 15. The Norwegian lives next to the blue house.*)
  poss1 = Position[t, "Norwegian"];
  poss2 = Position[t, "Blue"];
  poss = Tuples[{poss1, poss2}];
  poss = Select[poss, #[[1, 2]] + 1 == #[[2, 2]] \[Or] #[[1, 2]] - 1 == #[[2, 2]] &]\[Transpose];
  delpos = Complement[poss1, poss[[1]]];
  t = ReplacePart[t, delpos -> Null];
  delpos = Complement[poss2, poss[[2]]];
  t = ReplacePart[t, delpos -> Null];

  (* 5. The green house is immediately to the left of the white house. *)
  poss1 = Position[t, "Green"];
  poss2 = Position[t, "White"];
  poss = Tuples[{poss1, poss2}];
  poss = Select[poss, #[[1, 2]] + 1 == #[[2, 2]] &]\[Transpose];
  delpos = Complement[poss1, poss[[1]]];
  t = ReplacePart[t, delpos -> Null];
  delpos = Complement[poss2, poss[[2]]];
  t = ReplacePart[t, delpos -> Null];

  (*11. The man who smokes Blend lives in the house next to the house with cats.*)
  poss1 = Position[t, "Blend"];
  poss2 = Position[t, "Cats"];
  poss = Tuples[{poss1, poss2}];
  poss = Select[poss, #[[1, 2]] + 1 == #[[2, 2]] \[Or] #[[1, 2]] - 1 == #[[2, 2]] &]\[Transpose];
  delpos = Complement[poss1, poss[[1]]];
  t = ReplacePart[t, delpos -> Null];
  delpos = Complement[poss2, poss[[2]]];
  t = ReplacePart[t, delpos -> Null];

  (* 12. In a house next to the house where they have a horse, they smoke Dunhill. *)
  poss1 = Position[t, "Horse"];
  poss2 = Position[t, "Dunhill"];
  poss = Tuples[{poss1, poss2}];
  poss = Select[poss, #[[1, 2]] + 1 == #[[2, 2]] \[Or] #[[1, 2]] - 1 == #[[2, 2]] &]\[Transpose];
  delpos = Complement[poss1, poss[[1]]];
  t = ReplacePart[t, delpos -> Null];
  delpos = Complement[poss2, poss[[2]]];
  t = ReplacePart[t, delpos -> Null];

  (* 16. They drink water in a house next to the house where they smoke Blend. *)
  poss1 = Position[t, "Water"];
  poss2 = Position[t, "Blend"];
  poss = Tuples[{poss1, poss2}];
  poss = Select[poss, #[[1, 2]] + 1 == #[[2, 2]] \[Or] #[[1, 2]] - 1 == #[[2, 2]] &]\[Transpose];
  delpos = Complement[poss1, poss[[1]]];
  t = ReplacePart[t, delpos -> Null];
  delpos = Complement[poss2, poss[[2]]];
  t = ReplacePart[t, delpos -> Null];

  (*General rule 1 in a line => cross out vertical and horizontal lines*)
  (* 1 in a row*)
  vals = Select[t, Count[#, Null] == 4 &];
  vals = DeleteCases[Flatten[vals], Null];
  poss = Flatten[Position[t, #] & /@ vals, 1];
  delpos = With[{r = First[#], c = Last[#]}, {#, c} & /@ (Range[-4, 0] + Ceiling[r, 5])] & /@ poss; (*delete in columns*)
  delpos = Flatten[MapThread[DeleteCases, {delpos, poss}], 1];
  t = ReplacePart[t, delpos -> Null];

  (* 1 in a column*)
  sets = Flatten[Table[{i + k*5, j}, {k, 0, 4}, {j, 1, 5}, {i, 1, 5}],1];
  sets = {#, Extract[t, #]} & /@ sets;
  sets = Select[sets, Count[#[[2]], Null] == 4 &];
  sets = Flatten[Transpose /@ sets, 1];
  sets = DeleteCases[sets, {{_, _}, Null}];
  delpos = sets[[All, 1]];(*delete in rows*)
  delpos = With[{r = First[#], c = Last[#]}, {r, #} & /@ (DeleteCases[Range[5], c])] & /@ delpos;
  delpos = Flatten[delpos, 1];
  t = ReplacePart[t, delpos -> Null];

  t
]
colors = {"Blue", "Green", "Red", "White", "Yellow"};
nationality = {"Dane", "English", "German", "Norwegian", "Swede"};
beverage = {"Beer", "Coffee", "Milk", "Tea", "Water"};
animal = {"Birds", "Cats", "Dog", "Horse", "Zebra"};
smoke = {"Blend", "Blue Master", "Dunhill", "Pall Mall", "Prince"};
vals = {colors, nationality, beverage, animal, smoke};
bigtable = Join @@ (ConstantArray[#, 5]\[Transpose] & /@ vals);

bigtable = FixedPoint[FilterPuzzle, bigtable];
TableForm[DeleteCases[bigtable\[Transpose], Null, \[Infinity]], TableHeadings -> {Range[5], None}]
