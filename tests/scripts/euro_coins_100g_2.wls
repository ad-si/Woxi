
(* Euro coin weights in grams *)
(* 1 cent: 2.30 g, 2 cent: 3.06 g, 5 cent: 3.92 g,
   10 cent: 4.10 g, 20 cent: 5.74 g, 50 cent: 7.80 g,
   1 euro: 7.50 g, 2 euro: 8.50 g *)

coinNames = {"1 cent", "2 cent", "5 cent", "10 cent", "20 cent", "50 cent", "1 euro", "2 euro"};
coinWeights = {230, 306, 392, 410, 574, 780, 750, 850}; (* in hundredths of grams to use integers *)
target = 10000; (* 100.00 g in hundredths *)

(* Find a combination using integer linear programming *)
(* We want non-negative integers n_i such that sum(n_i * w_i) = 10000 *)
(* Minimize total number of coins *)

vars = Array[n, 8];
constraints = Join[
  {vars . coinWeights == target},
  Thread[vars >= 0]
];

sol = Minimize[{Total[vars], constraints}, vars, Integers];
Print[sol]
