(* Roll 5 dice: random integers between 1 and 6. Woxi's and wolframscript's
   RNG streams differ (even with the same seed), so print stream-independent
   properties of the rolls rather than the values themselves. *)
rolls = RandomInteger[{1, 6}, 5];
Print[Length[rolls]]
Print[AllTrue[rolls, IntegerQ]]
Print[AllTrue[rolls, 1 <= # <= 6 &]]
