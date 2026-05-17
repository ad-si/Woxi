Knapsack[shop_, capacity_] :=  Block[{sortedTable, overN, overW, output},
 sortedTable = SortBy[{#1, #2, #3, #3/#2} & @@@ shop, -#[[4]] &];
 overN = Position[Accumulate[sortedTable[[1 ;;, 2]]], a_ /; a > capacity, 1,1][[1, 1]];
 overW = Accumulate[sortedTable[[1 ;;, 2]]][[overN]] - capacity;

 output = Reverse@sortedTable[[Ordering[sortedTable[[1 ;;, 4]], -overN]]];
 output[[-1, 2]] = output[[-1, 2]] - overW;
 output[[-1, 3]] = output[[-1, 2]] output[[-1, 4]];
 Append[output[[1 ;;, 1 ;; 3]], {"Total",Sequence @@ Total[output[[1 ;;, 2 ;; 3]]]}]]


(* ========== *)

weightPriceTable =
{{"beef", 3.8, 36}, {"pork", 5.4, 43}, {"ham", 3.6, 90}, {"greaves", 2.4, 45}, {"flitch", 4., 30},
{"brawn", 2.5, 56}, {"welt", 3.7, 67}, {"salami", 3., 95}, {"sausage",  5.9, 98}};
carryCapacity = 15;
Knapsack[weightPriceTable, carryCapacity] // Grid

salami	3.	95
ham	3.6	90
brawn	2.5	56
greaves	2.4	45
welt	3.5	63.3784
Total	15.	349.378
