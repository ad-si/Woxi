smithQ[n_] := Not[PrimeQ[n]] &&
            Total[IntegerDigits[n]] == Total[IntegerDigits /@ Flatten[ConstantArray @@@ FactorInteger[n]],2];

Select[Range[2, 10000], smithQ]
