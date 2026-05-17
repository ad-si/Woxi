ListDensityPlot[
 Table[Table[
   FromDigits[BitXor[IntegerDigits[x, 2, 8], IntegerDigits[y, 2, 8]],
    2], {x, 0, 255}], {y, 0, 255}]]


(* ========== *)

ArrayPlot[Array[BitXor, {511, 511}]]
