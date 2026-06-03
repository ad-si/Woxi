a = "/* header */\nfunction f() {\n  x = 1; /* set x */\n}\n/* trailing\n   comment */\n";
Print[StringReplace[a, "/*" ~~ Shortest[___] ~~ "*/" -> ""]]
