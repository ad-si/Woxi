Complement[Range[100],
  Select[6 #[[1]] + 9 #[[2]] + 20 #[[3]] & /@
    Tuples[Range[0, 17], 3], # < 101 &]][[-1]]
