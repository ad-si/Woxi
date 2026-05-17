choices={{"aleph", 1/5},{"beth", 1/6},{"gimel", 1/7},{"daleth", 1/8},{"he", 1/9},{"waw", 1/10},{"zayin", 1/11},{"heth", 1759/27720}};
data=RandomChoice[choices[[All,2]]->choices[[All,1]],10^6];


(* ========== *)

Grid[{#[[1]],N[Count[data,#[[1]]]/10^6],N[#[[2]]]}&/@choices]
