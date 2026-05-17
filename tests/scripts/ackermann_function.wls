$RecursionLimit=Infinity
Ackermann1[m_,n_]:=
 If[m==0,n+1,
  If[ n==0,Ackermann1[m-1,1],
   Ackermann1[m-1,Ackermann1[m,n-1]]
  ]
 ]

 Ackermann2[0,n_]:=n+1;
 Ackermann2[m_,0]:=Ackermann1[m-1,1];
 Ackermann2[m_,n_]:=Ackermann1[m-1,Ackermann1[m,n-1]]


