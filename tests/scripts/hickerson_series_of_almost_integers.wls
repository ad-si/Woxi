h[n_] = n!/(2 (Log[2])^(n + 1));
firstdecimal[x_] := Floor[10 x] - 10 Floor[x];
almostIntegerQ[x_] := firstdecimal[x] == 0 || firstdecimal[x] == 9;
Table[{i, AccountingForm[N[h[i], 50], {Infinity, 5}], almostIntegerQ[N[h[i], 50]]}, {i, 1, 17}] // TableForm
