(* Warm-up shim (not part of the RosettaCode solution): the first calendar
   computation in a wolframscript session emits a one-time CalendarData init
   message to stdout (headless, no front-end). Triggering it here inside a
   Print-swallowing Block keeps both engines' stdout identical. Woxi emits
   nothing for this line. See find_the_last_sunday_of_each_month for details. *)
Block[{Print = Null &}, DayRange[{2000, 1, 1}, {2000, 1, 1}, Friday]];

LastFridays[year_] :=
 Table[Last@DayRange[{year, i},
    DatePlus[{year, i}, {{1, "Month"}, {-1, "Day"}}], Friday], {i, 12}]
Print[LastFridays[2012]]
