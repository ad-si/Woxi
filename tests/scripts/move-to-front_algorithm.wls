mtf[word_]:=Module[{f,f2,p,q},
   f[{output_,symList_},next_]:=Module[{index},index=Position[symList,next][[1,1]]-1;
       {output~Append~index,Prepend[Delete[symList,index+1],next]}];
   p=Fold[f,{{},CharacterRange["a","z"]},Characters[ToString[word]]][[1]];
   f2[{output_,symList_},next_]:=Module[{index},index=symList[[next+1]];
       {output~Append~index,Prepend[DeleteCases[symList,ToString[index]],index]}];
   q=Fold[f2,{{},CharacterRange["a","z"]},p][[1]];
   Print["'", word,"' encodes to: ",p, " - "  ,p," decodes to: '",StringJoin@q,"' - Input equals Output: " ,word===StringJoin@q];]


(* ========== *)

mtf /@ {"broood", "bananaaa", "hiphophiphop"}
