(* The RosettaCode task uses `Run["tput smcup/rmcup"]` and `Run["echo …"]`,
   but tput's output depends on the environment (TERM, terminfo) and on
   whether the engine's `Run` child inherits the captured stdout, so no
   single snapshot can be stable. Print the xterm alternate-screen
   sequences directly instead. *)
Print["\033[?1049h"]  (* Save the display *)
Print["Hello"]
Pause[5]       (* Wait five seconds *)
Print["\033[?1049l"]
