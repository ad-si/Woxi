Options[fn]={Add->False,Offset-> 0};
Print[fn[x_,y_,OptionsPattern[]]:=If[OptionValue[Add]==True,x+y+OptionValue[Offset],{x,y,OptionValue[Offset]}]];
Print[fn[3,4,{Add->True,Offset->2}]];
Print[fn[3,4,{Offset->2,Add->True}]];
