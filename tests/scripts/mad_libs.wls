(* The template text and the answers would normally come from
   InputString[]; both are hardcoded here. *)
text = "<name> went to the <place> to buy a <noun>.";
answers = <|
   "<name>" -> "Alice",
   "<place>" -> "market",
   "<noun>" -> "book"|>;
Print[StringReplace[text, Normal[answers]]];
