a = {1, 2, 3, 4, 5}
Print[Plus @@ a]
Print[Apply[Plus, a]]
Print[Total[a]]
Print[Total@a]
Print[a // Total]
Print[Sum[a[[i]], {i, 1, Length[a]}]]
Print[Sum[i, {i, a}]]

a = {1, 2, 3, 4, 5}
Print[Times @@ a]
Print[Apply[Times, a]]
Print[Product[a[[i]], {i, 1, Length[a]}]]
Print[Product[i, {i, a}]]
