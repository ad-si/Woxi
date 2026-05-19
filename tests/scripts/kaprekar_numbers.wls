KaprekarQ[1] := True
KaprekarQ[n_Integer] := Module[{data = IntegerDigits[n^2], last = False, i = 1},
  While[i < Length[data] && FromDigits[data[[i + 1 ;;]]] =!= 0 && Not[last],
    last = FromDigits[data[[;; i]]] + FromDigits[data[[i + 1 ;;]]] == n;
    i++];
  last]
Print[Select[Range[10000], KaprekarQ]]
