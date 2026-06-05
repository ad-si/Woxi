ranks = {"Ace", 2, 3, 4, 5, 6, 7, 8, 9, 10, "Jack", "Queen", "King"};
suits = {"Diamonds", "Clubs", "Hearts", "Spades"};
deck = Tuples[{ranks, suits}];
(* `DeckShuffle[d_] := RandomSample[d, Length@d]` normally shuffles the deck
   randomly. Hardcoded to the unshuffled order here so the snapshot is
   deterministic. *)
Print["Cards in deck: ", Length[deck]]
Print["Deal three: ", deck[[1 ;; 3]]]
