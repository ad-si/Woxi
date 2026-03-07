
(* Euro coin weights in grams *)
(* 1 cent: 2.30 g, 2 cent: 3.06 g, 5 cent: 3.92 g, 10 cent: 4.10 g, 
   20 cent: 5.74 g, 50 cent: 7.80 g, 1 euro: 7.50 g, 2 euro: 8.50 g *)

coinNames = {"1 cent", "2 cent", "5 cent", "10 cent", "20 cent", "50 cent", "1 euro", "2 euro"};
coinWeights = {230, 306, 392, 410, 574, 780, 750, 850}; (* in hundredths of grams to use integers *)
target = 10000; (* 100.00 g in hundredths *)

(* Find combinations that sum to exactly 100 g *)
(* We'll look for solutions with a reasonable number of coins *)
(* Using integer linear programming or brute force for small counts *)

(* Let's try to find solutions using IntegerPartitions-like approach *)
(* n1*230 + n2*306 + n3*392 + n4*410 + n5*574 + n6*780 + n7*750 + n8*850 = 10000 *)

(* Let's find all non-negative integer solutions with total coins <= say 20 *)
maxCoins = 50;
solutions = {};

(* Use Reduce or FindInstance *)
vars = {n1, n2, n3, n4, n5, n6, n7, n8};
constraints = {
  230 n1 + 306 n2 + 392 n3 + 410 n4 + 574 n5 + 780 n6 + 750 n7 + 850 n8 == 10000,
  n1 >= 0, n2 >= 0, n3 >= 0, n4 >= 0, n5 >= 0, n6 >= 0, n7 >= 0, n8 >= 0
};

(* Find several instances *)
sols = FindInstance[constraints, vars, Integers, 50]
