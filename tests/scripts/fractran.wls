fractionlist = {17/91, 78/85, 19/51, 23/38, 29/33, 77/29, 95/23, 77/19, 1/17, 11/13, 13/11, 15/14, 15/2, 55/1};
n = 2;
steplimit = 20;
j = 0;
break = False;
While[break == False && j <= steplimit,
 newlist = n fractionlist;
 isintegerlist = IntegerQ[#] & /@ newlist;
 truepositions = Position[isintegerlist, True];
 If[Length[truepositions] == 0,
  break = True,
  Print[ToString[j] <> ": " <> ToString[n]];
  n = newlist[[truepositions[[1, 1]]]]; j++;
  ]
 ]
