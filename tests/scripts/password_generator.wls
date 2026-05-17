(* Length is the Length of the password, num is the number you want, \
and similar=1 if you want similar characters, 0 if not. True and \
False, should work in place of 1/0 *)
pwgen[length_Integer: 5, num_Integer: 1, similars_Boolean: 1] :=
 pwgenerator[length, num, similars] =
  Module[{list, valid, validchars, similar, k, j, password, pwlist,
    lcase, ucase, digits, spec, s}, lcase = Alphabet[];
   ucase = Capitalize[lcase];
   digits = Range[0, 9];
   spec = StringPartition["!:#$%\'()*+,-./:;>=<?@{}[]^_|~", 1];
   validchars = Flatten[Union[lcase, ucase, digits, spec]];
   similar = StringPartition["Il10O5S2Z", 1];
   list = {};

   Table[valid = 0;
    While[valid == 0,
     For[j = 0; k = {};, j < length, j++,
      AppendTo[k, RandomInteger[{1, Length[validchars]}]]];
     k = Flatten[k];
     password = validchars[[k]];
     Which[(Intersection[password, similar] >= 1 && similars == 0 ),
      valid = 0;, (Intersection[password, similar] == 0 &&
        similars == 1),
      valid = 0;, (Intersection[password, similar] == 0 &&
        similars == 0 ), valid = 1;
      Return[password], (Intersection[password, similar] >= 1 &&
        similars == 1), valid = 1; Return[password]];
     ], {num}];


   ]
