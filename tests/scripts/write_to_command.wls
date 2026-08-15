(* A file name starting with `!` names an external command.
   The command's own output goes to the terminal, so these examples
   redirect it into a file and read that back. *)

stream = OpenWrite["!sort > sorted.txt"];
WriteString[stream, "gamma\nalpha\nbeta\n"];
Close[stream];
Print[ReadList["sorted.txt", String]];

Put[2 + 3, "!cat > put.txt"];
Print[ReadString["put.txt"]];

Print[Import["!printf 'a,b\\n1,2\\n'", "CSV"]];
Print[Get["!echo 6*7"]];
Print[Find["!printf 'alpha\\nbeta\\n'", "bet"]];
