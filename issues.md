tests/interpreter_tests/calculus.rs:2034
FAIL #1349: LaplaceTransform[BesselJ[0, t], t, s]
  Woxi:    LaplaceTransform[BesselJ[0, t], t, s]
  Wolfram: 1/Sqrt[1 + s^2]

tests/interpreter_tests/calculus.rs:2314
FAIL #1380: Wronskian[{Sin[x], Cos[x]}, x]
  Woxi:    -Cos[x]^2 - Sin[x]^2
  Wolfram: -1

tests/interpreter_tests/calculus.rs:2436
FAIL #1393: TrigToExp[Cosh[x]]
  Woxi:    1/2*1/E^x + E^x/2
  Wolfram: 1/(2*E^x) + E^x/2

tests/interpreter_tests/math/special_functions.rs:3045
FAIL #4019: HypergeometricPFQ[{1, 2}, {3}, 0.5]
  Woxi:    1.545177444479561
  Wolfram: 1.5451774444795618

tests/interpreter_tests/math/special_functions.rs:3069
FAIL #4022: N[HypergeometricPFQ[{1/2}, {3/2}, -1]]
  Woxi:    0.746824132812427
  Wolfram: 0.7468241328124269

tests/interpreter_tests/math/special_functions.rs:3077
FAIL #4023: N[HypergeometricPFQ[{1, 2}, {3}, 1/2]]
  Woxi:    1.545177444479561
  Wolfram: 1.5451774444795623

tests/interpreter_tests/math/special_functions.rs:3108
FAIL #4028: RiemannR[10.]
  Woxi:    4.564583141005088
  Wolfram: 4.564583141005087

tests/interpreter_tests/math/special_functions.rs:3113
FAIL #4029: RiemannR[100.]
  Woxi:    25.661633266924188
  Wolfram: 25.66163326692419

tests/interpreter_tests/math/special_functions.rs:3118
FAIL #4030: RiemannR[1000.]
  Woxi:    168.35944628116727
  Wolfram: 168.3594462811673

tests/interpreter_tests/math/special_functions.rs:3124
FAIL #4031: N[RiemannR[1000000]]
  Woxi:    78527.39942912769
  Wolfram: 78527.39942912766

tests/interpreter_tests/syntax.rs:1298
FAIL #4724: Attributes[Plot3D]
  Woxi:    {HoldAll, Protected, ReadProtected}
  Wolfram: {Protected, ReadProtected}

tests/interpreter_tests/syntax.rs:1549
FAIL #4763: TraditionalForm[x + y]
  Woxi:    TraditionalForm[x + y]
  Wolfram: DisplayForm[FormBox[RowBox[{x, +, y}], TraditionalForm]]

tests/interpreter_tests/syntax.rs:1565
FAIL #4765: TraditionalForm[1 + 2]
  Woxi:    TraditionalForm[3]
  Wolfram: DisplayForm[FormBox[3, TraditionalForm]]

tests/interpreter_tests/syntax.rs:1573
FAIL #4766: TraditionalForm[Sin[Pi/4]]
  Woxi:    TraditionalForm[1/Sqrt[2]]
  Wolfram: DisplayForm[FormBox[FractionBox[1, SqrtBox[2]], TraditionalForm]]

tests/interpreter_tests/syntax.rs:4287
FAIL #5133: TwoWayRule[a, b]
  Woxi:    TwoWayRule[a, b]
  Wolfram: a <-> b

tests/interpreter_tests/syntax.rs:4292
FAIL #5134: TwoWayRule[1, 2]
  Woxi:    TwoWayRule[1, 2]
  Wolfram: 1 <-> 2

tests/interpreter_tests/syntax.rs:4347
FAIL #5143: Around[5, 0.3]
  Woxi:    Around[5, 0.3]
  Wolfram: Around[5., 0.3]

tests/interpreter_tests/syntax.rs:4497
FAIL #5164: RotationTransform[x]
  Woxi:    RotationTransform[x]
  Wolfram: TransformationFunction[{{Cos[x], -Sin[x], 0}, {Sin[x], Cos[x], 0}, {0, 0, 1}}]
