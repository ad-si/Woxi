blocks = Partition[
  Characters[ToLowerCase["BOXKDQCPNAGTRETGQDFSJWHUVIANOBERFSLYPCZM"]],
  2
];

ClearAll[DoStep, ABCBlockQ]

DoStep[chars_List, blcks_List, chosen_List] := Module[{opts},
  If[chars =!= {},
    opts = Select[blcks, MemberQ[#, First[chars]] &];
    {Rest[chars], DeleteCases[blcks, #, 1, 1], Append[chosen, #]} & /@ opts
  ,
    {{chars, blcks, chosen}}
  ]
]

DoStep[opts_List] := Flatten[DoStep @@@ opts, 1]

ABCBlockQ[str_String] := (
  FixedPoint[DoStep, {{Characters[ToLowerCase[str]], blocks, {}}}] =!= {}
)

Print[ABCBlockQ["A"]]
Print[ABCBlockQ["BARK"]]
Print[ABCBlockQ["BOOK"]]
Print[ABCBlockQ["TREAT"]]
Print[ABCBlockQ["COMMON"]]
Print[ABCBlockQ["SQUAD"]]
Print[ABCBlockQ["CONFUSE"]]
