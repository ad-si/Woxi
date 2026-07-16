(* The RosettaCode task uses `Run["tput civis/cvvis"]`, but tput's output
   depends on the environment (TERM, terminfo) and on whether the engine's
   `Run` child inherits the captured stdout, so no single snapshot can be
   stable. Print the DEC cursor-visibility sequences directly instead. *)
Print["\033[?25l"]  (* Cursor hidden *)
Pause[2]
Print["\033[?25h"]  (* Cursor visible *)
