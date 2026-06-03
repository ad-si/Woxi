(* Warm-up shim (not part of the RosettaCode solution):
   The first calendar computation in a wolframscript session lazily
   initializes the CalendarData subsystem, which fires a one-time progress
   notification via Print[PrintTemporary]. In a notebook that shows a
   temporary status cell that auto-removes; headless wolframscript has no
   front-end, so it degrades to a plain Print that leaks to stdout as the
   raw, unformatted `StringForm[Initializing `1` indices ...., CalendarData]`
   (StringForm only substitutes its slots when boxed by a front-end).
   That status line is not part of any result and Woxi never emits it, so it
   would otherwise make this script's stdout differ from wolframscript's.
   Running one throwaway calendar call inside `Block[{Print = Null &}, ...]`
   triggers the init and swallows its message; because the init is
   once-per-session, the real calls below then run cleanly. In Woxi this line
   simply produces no output, so both engines' stdout match byte-for-byte. *)
Block[{Print = Null &}, DayRange[{2000, 1, 1}, {2000, 1, 1}, Sunday]];

LastSundays[year_] :=
 Table[Last@DayRange[{year, i},
    DatePlus[{year, i}, {{1, "Month"}, {-1, "Day"}}], Sunday], {i, 12}]
Print[LastSundays[2013]]
