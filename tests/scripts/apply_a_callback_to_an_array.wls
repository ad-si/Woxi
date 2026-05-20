Print[(#*#)& /@ {1, 2, 3, 4}]
Print[Map[Function[#*#], {1, 2, 3, 4}]]
Print[Map[(#*#)&, {1, 2, 3, 4}]]
Print[Map[Function[w, w*w], {1, 2, 3, 4}]]
