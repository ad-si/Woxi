fibdata = Array[First@IntegerDigits@Fibonacci@# &, 1000];
Table[{d, N@Count[fibdata, d]/Length@fibdata, Log10[1. + 1/d]}, {d, 1,
    9}] // Grid
