shuffle[deck_] := Apply[Riffle, TakeDrop[deck, Length[deck]/2]];
shuffleCount[n_] := Block[{count=0}, NestWhile[shuffle, shuffle[Range[n]], (count++; OrderedQ[#] )&];count];
Map[shuffleCount, {8, 24, 52, 100, 1020, 1024, 10000}]
