MillerRabin[n_,k_]:=Module[{d=n-1,s=0,test=True},While[Mod[d,2]==0 ,d/=2 ;s++]
Do[
  a=RandomInteger[{2,n-1}]; x=PowerMod[a,d,n];
  If[x!=1,
   For[ r = 0, r < s, r++, If[x==n-1, Continue[]]; x = Mod[x*x, n]; ];
   If[ x != n-1, test=False ];
  ];
,{k}];
Print[test] ]


