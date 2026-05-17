list={{5,"Buzz"},{3,"Fizz"},{7,"Baxx"}};
runTo=(*LCM@@list[[All,1]]+1*)20;
Column@Table[
 Select[list,Mod[x,#[[1]]]==0&][[All,2]]/.{}->{x}
 ,{x,1,runTo}
]
