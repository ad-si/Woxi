CountryCodes={{"AL",28},{"AD",24},{"AT",20},{"AZ",28},{"BE",16},{"BH",22},{"BA",20},{"BR",29},{"BG",22},{"CR",21},{"HR",21},{"CY",28},{"CZ",24},{"DK",18},{"DO",28},{"EE",20},{"FO",18},{"FI",18},{"FR",27},{"GE",22},{"DE",22},{"GI",23},{"GR",27},{"GL",18},{"GT",28},{"HU",28},{"IS",26},{"IE",22},{"IL",23},{"IT",27},{"KZ",20},{"KW",30},{"LV",21},{"LB",28},{"LI",21},{"LT",20},{"LU",20},{"MK",19},{"MT",31},{"MR",27},{"MU",30},{"MC",27},{"MD",24},{"ME",22},{"NL",18},{"NO",15},{"PK",24},{"PS",29},{"PL",28},{"PT",25},{"RO",24},{"SM",27},{"SA",24},{"RS",22},{"SK",24},{"SI",19},{"ES",24},{"SE",24},{"CH",21},{"TN",24},{"TR",26},{"AE",23},{"GB",22},{"VG",24}};
ClearAll[IBANVerify]
IBANVerify[input_String]:=Module[{i,cc,rules},
 i=StringReplace[StringTrim[input],{" "->"","\t"->""}];
 cc=StringTake[i,2];
 If[MemberQ[CountryCodes[[All,1]],cc]
 ,
  cc=Select[CountryCodes,First[#]==cc&][[1,2]];
  If[cc==StringLength[i]
  ,
   i=StringRotateLeft[i,4];
   i=Characters[ToUpperCase[i]];
   rules=Rule@@@({CharacterRange["A","Z"],Range[10,35]}\[Transpose]);
   i=i/.rules;
   i=ToExpression/@i;
   i=FromDigits[Flatten[IntegerDigits/@i]];
   If[Mod[i,97]===1
   ,
    True
   ,
    False
   ]
  ,
   False
  ]
 ,
  False
 ]
]


(* ========== *)

IBANVerify["GB82 WEST 1234 5698 7654 32"]
IBANVerify["GB82 WEST 1234 5698 7654 323"]
IBANVerify["GB82 WEST 1234 5698 7654 31"]
IBANVerify["XX82 WEST 1234 5698 7654 323"]
