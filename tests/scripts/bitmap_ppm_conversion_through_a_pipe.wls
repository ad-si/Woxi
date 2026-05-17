convert[image_,out_]:=Module[{process=StartProcess[{
"wolfram","-noinit","-noprompt","-run",
"Export[FromCharacterCode["~~ToString[ToCharacterCode[out]]~~"],ImportString[StringRiffle[Table[InputString[],{4}],FromCharacterCode[10]],FromCharacterCode[{80,80,77}]]]"
}]},
WriteLine[process,image];
WriteLine[process,"Quit[]"];
];
