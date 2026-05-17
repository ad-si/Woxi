a = {1, 2, 3, 4, 5}
Plus @@ a
Apply[Plus, a]
Total[a]
Total@a
a // Total
Sum[a[[i]], {i, 1, Length[a]}]
Sum[i, {i, a}]


(* ========== *)

a = {1, 2, 3, 4, 5}
Times @@ a
Apply[Times, a]
Product[a[[i]], {i, 1, Length[a]}]
Product[i, {i, a}]
