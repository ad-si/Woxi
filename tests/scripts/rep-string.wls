RepStringQ[strin_String]:=StringCases[strin,StartOfString~~Repeated[x__,{2,\[Infinity]}]~~y___~~EndOfString/;StringMatchQ[x,StartOfString~~y~~___]:>x, Overlaps -> All]
