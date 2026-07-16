(* The RosettaCode task uses `Run["clear"]`, but the output of `clear`
   depends on the environment (TERM, ncurses version, whether stdout is a
   tty) and on whether the engine's `Run` child inherits the captured
   stdout, so no single snapshot can be stable. Print the ANSI
   home-and-clear sequence directly instead — deterministic in both
   engines and still clears the screen when run in a terminal. *)
Print["\033[H\033[2J"]
