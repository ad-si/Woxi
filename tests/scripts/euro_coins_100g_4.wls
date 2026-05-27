(* Euro coin weights in grams *)
coins = {
  "2\[Euro]" -> 8.50,
  "1\[Euro]" -> 7.50,
  "50c" -> 7.80,
  "20c" -> 5.74,
  "10c" -> 4.10,
  "5c" -> 3.92,
  "2c" -> 3.06,
  "1c" -> 2.30
};

weights = Values[coins];
names = Keys[coins];

(* Find the combination with fewest coins that sums to exactly 100g *)
(* We use IntegerPartitions-like approach via LinearProgramming or brute force *)
(* Let's use ILP: minimize total number of coins, subject to weight = 100 *)

(* Variables: n1..n8 (number of each coin type) *)
(* Minimize: sum of ni *)
(* Subject to: sum of wi*ni = 100, ni >= 0 integer *)

(* Since weights have decimals, multiply everything by 100 to work with integers *)
intWeights = Round[weights * 100];
target = 10000;

(* Use a bounded search - max coins of any type is about 100/min_weight ~ 44 *)
maxCoins = Ceiling[target / Min[intWeights]];

(* Let's try to find solutions using Reduce or similar *)
vars = Array[n, 8];
constraints = Join[
  {intWeights . vars == target},
  Thread[vars >= 0]
];

(* Find minimum total coins *)
result = Minimize[{Total[vars], And @@ constraints, Element[vars, Integers]}, vars];
Print[result]
