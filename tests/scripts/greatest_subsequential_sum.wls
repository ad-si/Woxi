Sequences[m_]:=Prepend[Flatten[Table[Partition[Range[m],n,1],{n,m}],1],{}]
MaximumSubsequence[x_List]:=Module[{sums},
 sums={x[[#]],Total[x[[#]]]}&/@Sequences[Length[x]];
 First[First[sums[[Ordering[sums,-1,#1[[2]]<#2[[2]]&]]]]]
]


