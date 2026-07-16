(* The RosettaCode task uses `Run["tput mr/me"]` and `Run["echo …"]`, but
   tput's output depends on the environment (TERM, terminfo) and on whether
   the engine's `Run` child inherits the captured stdout, so no single
   snapshot can be stable. Print the ANSI video-attribute sequences
   directly instead. *)
Print["\033[7mfoo"] (* is displayed in reverse mode *)
Print["\033[0mbar"]
