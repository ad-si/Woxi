exponentiation[x_,y_Integer]:=Which[y>0,Times@@ConstantArray[x,y],y==0,1,y<0,1/exponentiation[x,-y]]
CirclePlus[x_,y_Integer]:=exponentiation[x,y]


(* ========== *)

exponentiation[1.23,3]
exponentiation[4,0]
exponentiation[2.5,-2]
1.23\[CirclePlus]3
4\[CirclePlus]0
2.5\[CirclePlus]-2


(* ========== *)

1.86087
1
0.16
1.86087
1
0.16
