SumQ[u_] :=
  Head[u]===Plus

ProductQ[u_] :=
  Head[u]===Times

PowerQ[u_] :=
  Head[u]===Power

RationalQ[u_] :=
  If[ListQ[u],
    MapAnd[RationalQ,u],
  IntegerQ[u] || Head[u]===Rational]


PosQ[u_] :=
  If[PossibleZeroQ[u],
    False,
  If[NumericQ[u],
    If[NumberQ[u],
      If[PossibleZeroQ[Re[u]],
        Im[u]>0,
      Re[u]>0],
    Module[{v=N[u]},
    If[PossibleZeroQ[Re[v]],
      Im[v]>0,
    Re[v]>0]]],
(*If[ProductQ[u],
    PosQ[First[u]], *)
  Module[{v=Simplify[u]},
  If[NumericQ[v],
    PosQ[v],
  If[PowerQ[v] && IntegerQ[v[[2]]],
    PosQ[v[[1]]],
  If[ProductQ[v],
    If[RationalQ[First[v]],
      If[First[v]>0,
        PosQ[Rest[v]],
      NegQ[Rest[v]]],
    PosQ[First[v]]],
  If[SumQ[v],
    PosQ[First[v]],
  Not[MatchQ[v,-_]]]]]]]]]


NegQ[u_] :=
  If[PossibleZeroQ[u],
    False,
  Not[PosQ[u]]]


Print[NegQ[-a/b]]