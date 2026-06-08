(* varname and varvalue would normally come from InputString[]; hardcoded here. *)
varname = "myVar";
varvalue = "Hello";
ReleaseHold[
  Hold[Set["nameholder", "value"]] /.
   {"nameholder" -> Symbol[varname], "value" -> varvalue}];
Print[varname, " is now set to ", Symbol[varname]]
