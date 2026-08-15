stream = OpenRead["!printf 'alpha\\nbeta\\ngamma\\n'"];
While[True,
  line = ReadLine[stream];
  If[line === EndOfFile, Break[]];
  Print["Processed line: ", line];
];
Close[stream];
