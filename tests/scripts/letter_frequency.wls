Export["text.txt", "the quick brown fox jumps over the lazy dog", "Text"];
Print[Tally[Characters[Import["text.txt", "Text"]]]]
