EncodeBase[number_,base_]:=Module[{
  out={},
  rem,n=number,b=base
 },
 While[
  n!=0,
  {n,rem}={Floor[Divide[n,b]],Mod[n,b]};
  If[
   rem<0,
   n+=1;
   rem-=b
  ];
  PrependTo[out,rem]
 ];
 out
];
DecodeBase[list_,base_]:=Total[list*base^Range[Length[list]-1,0,-1]];

Print[EncodeNegBase[10,-2],"=",DecodeBase[EncodeNegBase[10,-2],-2]];
Print[EncodeNegBase[146,-3],"=",DecodeBase[EncodeNegBase[146,-3],-3]];
Print[EncodeNegBase[15,-10],"=",DecodeBase[EncodeNegBase[15,-10],-10]];


(* ========== *)

DecodeBase[ToCharacterCode[$Version], -126](*ascii 1-byte encoding*)
DecodeBase[ToCharacterCode[$Version], -(2^16 - 1)](*2-byte encoding*)
