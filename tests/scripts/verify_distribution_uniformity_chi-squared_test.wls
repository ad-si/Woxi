discreteUniformDistributionQ[data_, {min_Integer, max_Integer}, confLevel_: .05] :=
If[$VersionNumber >= 8,
  confLevel <= PearsonChiSquareTest[data, DiscreteUniformDistribution[{min, max}]],
  Block[{v, k = max - min, n = Length@data},
   v = (k + 1) (Plus @@ (((Length /@ Split[Sort@data]))^2))/n - n;
   GammaRegularized[k/2, 0, v/2] <= 1 - confLevel]]

discreteUniformDistributionQ[data_] :=discreteUniformDistributionQ[data, data[[Ordering[data][[{1, -1}]]]]]


(* ========== *)

uniformData = RandomInteger[10, 100];
nonUniformData = Total@RandomInteger[10, {5, 100}];


(* ========== *)

{discreteUniformDistributionQ[uniformData],discreteUniformDistributionQ[nonUniformData]}
