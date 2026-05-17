haystack = {"Zig","Zag","Wally","Ronald","Bush","Zig","Zag","Krusty","Charlie","Bush","Bozo"};
needle = "Zag";
first = Position[haystack,needle,1][[1,1]]
last = Position[haystack,needle,1][[-1,1]]
all = Position[haystack,needle,1][[All,1]]


(* ========== *)

2
7
{2,7}
