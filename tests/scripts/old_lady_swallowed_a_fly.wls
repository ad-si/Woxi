animals = {"fly", "spider", "bird", "cat", "dog", "goat", "cow",
   "horse"};
notes = {"", "That wiggled and jiggled and tickled inside her.\n",
   "How absurd, to swallow a bird.\n",
   "Imagine that. She swallowed a cat.\n",
   "What a hog to swallow a dog.\n",
   "She just opened her throat and swallowed that goat.\n",
   "I don't know how she swallowed that cow.\n",
   "She's dead, of course.",
   "I don't know why she swallowed that fly.\nPerhaps she'll die.\n\n\
"};
Print[StringJoin @@ ("There was an old lady who swallowed a " <>
       animals[[#]] <> ".\n" <> notes[[#]] <>
       If[# == 8, "",
        StringJoin @@ ("She swallowed the " <> animals[[#]] <>
              " to catch the " <> animals[[# - 1]] <> ".\n" & /@
            Range[#, 2, -1]) <> notes[[9]]] & /@ Range[8])];
