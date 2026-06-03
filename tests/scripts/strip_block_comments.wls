(* Strip /* ... */ block comments. Shortest wraps the whole sub-pattern so the
   match is non-greedy (equivalent to "/*"~~Shortest[___]~~"*/" for this task);
   the comment may span newlines. *)
a = "/* header */\nfunction f() {\n  x = 1; /* set x */\n}\n/* trailing\n   comment */\n";
Print[StringReplace[a, Shortest["/*" ~~ ___ ~~ "*/"] -> ""]]
