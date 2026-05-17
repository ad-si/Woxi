{pva,pwe,pvo}={3000,3/10,1/40};
{iva,iwe,ivo}={1800,2/10,3/200};
{gva,gwe,gvo}={2500,2,2/1000};
wemax=25;
vomax=1/4;
{pmax,imax,gmax}=Floor/@{Min[vomax/pvo,wemax/pwe],Min[vomax/ivo,wemax/iwe],Min[vomax/gvo,wemax/gwe]};

data=Flatten[Table[{{p,i,g}.{pva,iva,gva},{p,i,g}.{pwe,iwe,gwe},{p,i,g}.{pvo,ivo,gvo},{p,i,g}},{p,0,pmax},{i,0,imax},{g,0,gmax}],2];
data=Select[data,#[[2]]<=25&&#[[3]]<=1/4&];
First[SplitBy[Sort[data,First[#1]>First[#2]&],First]]


(* ========== *)




(* ========== *)

p:9 i:0 v:11
p:6 i:5 v:11
p:3 i:10 v:11
p:0 i:15 v:11
