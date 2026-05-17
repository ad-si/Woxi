makeList[sep_String]:=Block[
  {counter=0, makeItem},
  makeItem[item_String]:=ToString[++counter]<>sep<>item;
  makeItem /@ {"first", "second", "third"}
  ]
Scan[Print, makeList[". "]]
