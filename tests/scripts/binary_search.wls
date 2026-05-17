BinarySearchRecursive[x_List, val_, lo_, hi_] :=
 Module[{mid = lo + Round@((hi - lo)/2)},
  If[hi < lo, Return[-1]];
  Return[
   Which[x[[mid]] > val, BinarySearchRecursive[x, val, lo, mid - 1],
    x[[mid]] < val, BinarySearchRecursive[x, val, mid + 1, hi],
    True, mid]
   ];
  ]


(* ========== *)

BinarySearch[x_List, val_] := Module[{lo = 1, hi = Length@x, mid},
  While[lo <= hi,
   mid = lo + Round@((hi - lo)/2);
   Which[x[[mid]] > val, hi = mid - 1,
    x[[mid]] < val, lo = mid + 1,
    True, Return[mid]
    ];
   ];
  Return[-1];
  ]
