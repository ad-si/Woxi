RunLengthEncode[x_List]:=(Through[{First,Length}[#]]&)/@Split[x]
 LookAndSay[n_,d_:1]:=NestList[Flatten[Reverse/@RunLengthEncode[#]]&,{d},n-1]


(* ========== *)

FromDigits /@ LookAndSay[13] // Column
FromDigits /@ LookAndSay[13, 7] // Column
