a = {"foo", \[Pi], {<|
     "deep" -> {# +
         1 &, {{"Mathematica"}, {{"is"}, {"a"}}, {{{"cool"}}}, \
{{"programming"}, {"language!"}}}}|>}};
b = a;
a[[2]] -= 3;
a[[3, 1, 1, 1]] = #^2 &;
Print[a];
Print[b];
