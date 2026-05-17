Max@Select[FromDigits/@Rest@Flatten[Permutations/@Subsets[Range@9,9],1],And@@IntegerQ/@(#/IntegerDigits@#)&]
