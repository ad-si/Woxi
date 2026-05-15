obj[foo] = "This is foo.";
obj[bar] = "This is bar.";
obj[f_Symbol] := "What is " <> SymbolName[f] <> "?";
Print[obj@foo];
Print[obj@bar];
Print[obj@baz];
