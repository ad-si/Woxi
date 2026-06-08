Export["lines.txt",
  "line one\nline two\nline three\nline four\nline five\nline six\nline seven\nline eight",
  "Text"];
(* Read the first 7 lines; print the 7th. *)
lines = ReadList["lines.txt", String, 7];
Print[Last[lines]]
