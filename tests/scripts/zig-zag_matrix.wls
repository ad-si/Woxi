ZigZag[size_Integer/;size>0]:=Module[{empty=ConstantArray[0,{size,size}]},
 empty=ReplacePart[empty,{i_,j_}:>1/2 (i+j)^2-(i+j)/2-i (1-Mod[i+j,2])-j Mod[i+j,2]];
 ReplacePart[empty,{i_,j_}/;i+j>size+1:> size^2-tmp[[size-i+1,size-j+1]]-1]
]


