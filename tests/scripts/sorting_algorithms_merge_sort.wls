MergeSort[m_List] := Module[{middle},
  If[Length[m] >= 2,
   middle = Ceiling[Length[m]/2];
   Apply[mergeLists, Map[MergeSort, Partition[m, middle, middle, {1, 1}, {}]]],
   m]]
mergeLists[left_List, right_List] := Module[{leftIndex = 1, rightIndex = 1},
  Table[Which[
    leftIndex > Length[left], right[[rightIndex++]],
    rightIndex > Length[right], left[[leftIndex++]],
    left[[leftIndex]] <= right[[rightIndex]], left[[leftIndex++]],
    True, right[[rightIndex++]]], {Length[left] + Length[right]}]]
Print[MergeSort[{5, 3, 8, 1, 9, 2, 7, 4, 6}]]
