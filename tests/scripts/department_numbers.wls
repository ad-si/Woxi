Select[Permutations[Range[7], {3}], Total[#] == 12 && EvenQ[First[#]] &]
