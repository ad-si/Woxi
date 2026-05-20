ruleslower = Thread[# -> RotateLeft[#, 13]] &[CharacterRange["a", "z"]];
rulesupper = Thread[# -> RotateLeft[#, 13]] &[CharacterRange["A", "Z"]];
rules = Join[ruleslower, rulesupper];
text = "Hello World! Are you there!?";
text = StringReplace[text, rules];
Print[text];
text = StringReplace[text, rules];
Print[text];
