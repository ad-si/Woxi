string="In olden times when wishing still helped one, there lived a king whose daughters were all beautiful, but the youngest was so beautiful that the sun itself, which has seen so much, was astonished whenever it shone in her face.  Close by the king's castle lay a great dark forest, and under an old lime-tree in the forest was a well, and when the day was very warm, the king's child went out into the forest and sat down by the side of the cool fountain, and when she was bored she took a golden ball, and threw it up on high and caught it, and this ball was her favorite plaything.";

wordWrap[textWidth_, spaceWidth_, string_] := Module[{start, spaceLeft, masterString},
  spaceLeft=textWidth;
  start=1;
  masterString={};
  Do[
   If[i+1>Length@StringSplit@string
    ,
    p=StringSplit[string][[start;;i]];
    AppendTo[masterString, {StringJoin@@Riffle[p, StringJoin@ConstantArray[" ", spaceWidth]]}]
    ,
    If[StringLength[StringSplit@string][[i+1]]+spaceWidth>spaceLeft
      ,
      spaceLeft=textWidth-StringLength[StringSplit@string][[i]];
      start=i;
      AppendTo[masterString, {StringJoin@@Riffle[p, StringJoin@ConstantArray[" ", spaceWidth]]}]
      ,
      spaceLeft-=StringLength[StringSplit@string][[i]];
      spaceLeft-=spaceWidth;
      p=StringSplit[string][[start;;i]]
     ]
    ]
   ,
   {i, 1, Length@StringSplit@string}
  ];
  StringJoin@@Riffle[masterString, "\n"]
];

Print[worWrap[40, 1, string]]
