pangramQ[msg_]:=Complement[CharacterRange["a", "z"], Characters[ToLowerCase[msg]]]=== {}


(* ========== *)

pangramQ[msg_] :=
 Function[If[# === {}, Print["The string is a pangram!"],
    Print["The string is not a pangram. It's missing the letters " <>
      ToString[#]]]][
  Complement[CharacterRange["a", "z"], Characters[ToLowerCase[msg]]]]
