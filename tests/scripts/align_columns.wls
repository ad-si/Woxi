text = "Given$a$text$file$of$many$lines
are$delineated$by$a$single$char
write$a$program";
rows = StringSplit[#, "$"] & /@ StringSplit[text, "\n"];
width = Max[StringLength /@ Flatten[rows]];
Do[Print[StringJoin[(StringPadRight[#, width + 1]) & /@ row]], {row, rows}]
