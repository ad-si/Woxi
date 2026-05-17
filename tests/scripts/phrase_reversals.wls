phrase = "Rosetta Code Phrase Reversal";

reverseWords[phrase_String] :=
 StringJoin @@ Riffle[Reverse@StringSplit@phrase, " "]

reverseLetters[phrase_String] :=

 StringJoin @@
  Riffle[Map[StringJoin @@ Reverse[Characters@#] &,
    StringSplit@phrase], " "]

{phrase, reverseWords@phrase, reverseLetters@phrase,
  reverseWords@reverseLetters@phrase} // TableForm
