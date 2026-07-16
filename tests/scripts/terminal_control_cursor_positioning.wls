(* The RosettaCode task uses `Run["tput cup 6 3"]`, but tput's output
   depends on the environment (TERM, terminfo) and on whether the engine's
   `Run` child inherits the captured stdout, so no single snapshot can be
   stable. Print the ANSI cursor-position sequence directly instead. *)
Print["\033[7;4HHello"]
