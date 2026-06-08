Export["words.txt", "alpha beta gamma\ndelta epsilon", "Text"];
stream = OpenRead["words.txt"];
While[(w = Read[stream, Word]) =!= EndOfFile, Print[w]];
Close[stream];
