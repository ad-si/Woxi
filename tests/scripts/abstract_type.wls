(* Define an interface, Foo, which requires that the functions Foo, Bar, and Baz be defined *)
InterfaceFooQ[obj_] := ValueQ[Foo[obj]] && ValueQ[Bar[obj]] && ValueQ[Baz[obj]];
PrintFoo[obj_] := Print["Object ", obj, " does not implement interface Foo."];
PrintFoo[obj_?InterfaceFooQ] := Print[
   "Foo: ", Foo[obj], "\n",
   "Bar: ", Bar[obj], "\n",
   "Baz: ", Baz[obj], "\n"];

(* Extend all integers with Interface Foo *)
Foo[x_Integer] := Mod[x, 2];
Bar[x_Integer] := Mod[x, 3];
Baz[x_Integer] := Mod[x, 5];

(* Extend a particular string with Interface Foo *)
Foo["Qux"] = "foo";
Bar["Qux"] = "bar";
Baz["Qux"] = "baz";

(* Print a non-interface object *)
PrintFoo[{"Some", "List"}];
(* And for an integer *)
PrintFoo[8];
(* And for the specific string *)
PrintFoo["Qux"];
(* And finally a non-specific string *)
PrintFoo["foobarbaz"]
