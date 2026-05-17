hashJoin[table1_List,table1colindex_Integer,table2_List,table2colindex_Integer]:=Module[{h,f,t1,t2,tmp},
t1=If[table1colindex != 1,table1[[All,Prepend[Delete[Range@Length@table1[[1]],table1colindex],table1colindex]]],table1];
t2=If[table2colindex != 1, table2[[All,Prepend[Delete[Range@Length@table2[[1]],table2colindex],table2colindex]]],table2];

If[Length[t1]>Length[t2],tmp=t1;t1=t2;t2=tmp;];
h= GroupBy[t1,First];
f[{a_,b_List}]:={a,#}&/@b;
Partition[Flatten[Map[f,{#[[2;;]],h[#[[1]]]}&/@t2
]],Length[t1[[1]]]+Length[t2[[1]]]-1]
];
