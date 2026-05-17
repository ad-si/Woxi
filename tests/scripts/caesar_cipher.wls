cypher[mesg_String,n_Integer]:=StringReplace[mesg,Flatten[Thread[Rule[#,RotateLeft[#,n]]]&/@CharacterRange@@@{{"a","z"},{"A","Z"}}]]
