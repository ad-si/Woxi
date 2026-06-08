(* Create a small file in the working directory, then read it back whole. *)
Export["sample.txt", "First line\nSecond line\nThird line", "Text"];
Print[Import["sample.txt", "String"]]
