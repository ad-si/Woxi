(* Combine Euro coins to create 100g *)

(* Let's find integer combinations that sum to exactly 100 g *)
(* Using integer linear programming / solving *)
(* Variables: n1c, n2c, n5c, n10c, n20c, n50c, n1e, n2e *)
(* We want: 2.30*n1c + 3.06*n2c + 3.92*n5c + 4.10*n10c + 5.74*n20c + 7.80*n50c + 7.50*n1e + 8.50*n2e = 100 *)

(* Multiply everything by 100 to work with integers *)
(* 230*n1c + 306*n2c + 392*n5c + 410*n10c + 574*n20c + 780*n50c + 750*n1e + 850*n2e = 10000 *)

names = {"1 cent", "2 cent", "5 cent", "10 cent", "20 cent", "50 cent", "1 euro", "2 euro"};
weights = {230, 306, 392, 410, 574, 780, 750, 850};

(* Find solutions with small total number of coins *)
(* Try to minimize total coins *)
vars = {c1, c2, c5, c10, c20, c50, e1, e2};

(* Use integer linear programming to minimize total number of coins *)
result1 = Minimize[
  {
    Total[vars],
    vars . weights == 10000
      && And @@ (# >= 0 & /@ vars)
      && And @@ (Element[#, Integers] & /@ vars)
  },
  vars
];
Print[result1];

result2 = Minimize[
  {
    c1 + c2 + c5 + c10 + c20 + c50 + e1 + e2,
    230*c1 + 306*c2 + 392*c5 + 410*c10 + 574*c20 + 780*c50 + 750*e1 + 850*e2 == 10000
      && c1 >= 0 && c2 >= 0 && c5 >= 0 && c10 >= 0 && c20 >= 0 && c50 >= 0 && e1 >= 0 && e2 >= 0
      && Element[c1, Integers] && Element[c2, Integers] && Element[c5, Integers] && Element[c10, Integers] && Element[c20, Integers] && Element[c50, Integers] && Element[e1, Integers] && Element[e2, Integers]
  },
  {c1, c2, c5, c10, c20, c50, e1, e2}
];
Print[result2];
