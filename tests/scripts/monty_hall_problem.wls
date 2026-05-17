montyHall[nGames_] :=
    Module[{r, winningDoors, firstChoices, nStayWins, nSwitchWins, s},
         r := RandomInteger[{1, 3}, nGames];
         winningDoors = r;
         firstChoices = r;
         nStayWins =  Count[Transpose[{winningDoors, firstChoices}], {d_, d_}];
         nSwitchWins = nGames - nStayWins;

    Grid[{{"Strategy", "Wins", "Win %"}, {"Stay", Row[{nStayWins, "/", nGames}], s=N[100 nStayWins/nGames]},
          {"Switch", Row[{nSwitchWins, "/", nGames}], 100 - s}},  Frame -> All]]


(* ========== *)

montyHall[100000]
