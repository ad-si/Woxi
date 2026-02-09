a = "5^3^2";
Print[a <> " = " <> ToString[ToExpression[a]]]
b = "(5^3)^2";
Print[b <> " = " <> ToString[ToExpression[b]]]
c = "5^(3^2)";
Print[c <> " = " <> ToString[ToExpression[c]]]
