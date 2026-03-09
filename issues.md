tests/interpreter_tests/list.rs:3357
FAIL #2849: PDF[UniformDistribution[{0, 1}], x]
  Woxi:    Piecewise[{{1, Inequality[0, LessEqual, x, LessEqual, 1]}}, 0]
  Wolfram: Piecewise[{{1, 0 <= x <= 1}}, 0]

tests/interpreter_tests/list.rs:3365
FAIL #2850: PDF[UniformDistribution[{a, b}], x]
  Woxi:    Piecewise[{{(-a + b)^(-1), Inequality[a, LessEqual, x, LessEqual, b]}}, 0]
  Wolfram: Piecewise[{{(-a + b)^(-1), a <= x <= b}}, 0]

tests/interpreter_tests/list.rs:3373
FAIL #2851: PDF[UniformDistribution[], x]
  Woxi:    Piecewise[{{1, Inequality[0, LessEqual, x, LessEqual, 1]}}, 0]
  Wolfram: Piecewise[{{1, 0 <= x <= 1}}, 0]

tests/interpreter_tests/statistics.rs:638
FAIL #4375: CDF[UniformDistribution[{0, 1}], x]
  Woxi:    Piecewise[{{x, Inequality[0, LessEqual, x, LessEqual, 1]}, {1, x > 1}}, 0]
  Wolfram: Piecewise[{{x, 0 <= x <= 1}, {1, x > 1}}, 0]
