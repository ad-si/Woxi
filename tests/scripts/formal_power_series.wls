cos = Series[Cos[x], {x, 0, 10}];
sin = Series[Sin[x], {x, 0, 8}];
sin - Integrate[cos, x]
