a = <|2 -> "string", "sometext" -> 23|>;

Print["Keys: ", Keys[a]]
Print["Values: ", Values[a]]
KeyValueMap[Print["  ", #1, " -> ", #2] &, a]
