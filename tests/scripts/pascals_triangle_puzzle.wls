b+c==a
d+e==b
e+f==c
g+h==d
h+i==e
i+j==f
l+X==g
l+Y==h
n+Y==i
n+Z==j
X+Z==Y


(* ========== *)

a->151
d->40
l->11
n->4


(* ========== *)

eqs={a==b+c,d+e==b,e+f==c,g+h==d,h+i==e,i+j==f,l+X==g,l+Y==h,n+Y==i,n+Z==j,Y==X+Z};
knowns={a->151,d->40,l->11,n->4};
Solve[eqs/.knowns,{b,c,e,f,g,h,i,j,X,Y,Z}]


(* ========== *)




(* ========== *)

151
			81		70
		40		41		29
	16		24		17		12
5		11		13		4		8


(* ========== *)

triangle[n_, m_] :=  Nest[MovingMap[Total, #, 1] &, {x, 11, y, 4, z}, n - 1][[m]]
Solve[{triangle[3, 1] == 40, triangle[5, 1] == 151, y == x + z}, {x, y, z}]


(* ========== *)


