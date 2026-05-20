myPermanent[m_List] :=
    With[{v = Array[x, Length[m]]},
      Coefficient[Times @@ (m.v), Times @@ v]
  ]
Print[myPermanent[{{1, 2}, {3, 4}}]]
Print[myPermanent[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]]
Print[Det[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]]
