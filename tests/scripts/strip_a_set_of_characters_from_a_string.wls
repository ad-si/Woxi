Print[stripchars[a_,b_]:=StringReplace[a,(#->"")&/@Characters[b]]];
Print[stripchars["She was a soul stripper. She took my heart!","aei"]];
