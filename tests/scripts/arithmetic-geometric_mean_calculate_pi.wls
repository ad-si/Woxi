piCalc[n_,precision_]:=($precision=precision;4*a[n]^2)/(1-Sum[2^(1+k)*(a[k]^2-b[k]^2),{k,1,n}])
a[h_]:=(a[h]=(N[#,$precision]&@a[h-1]+b[h-1])/2)
b[h_]:=(b[h]=N[#,$precision]&@Sqrt[a[h-1] b[h-1]])
a[0]=1;
b[0]=1/Sqrt[2];

N[Pi, 1000000] - piCalc[10, 5]
0.*10^-996
