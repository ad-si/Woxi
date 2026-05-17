quibble[words___] :=
    ToString@{StringJoin@@
        Replace[Riffle[{words}, ", "],
            {most__, ", ", last_} -> {most, " and ", last}]}
