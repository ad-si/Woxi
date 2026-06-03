p = Position[#1, #2][[1, 1]] &;
Print[Select[
  Permutations[{"Baker", "Cooper", "Fletcher", "Miller", "Smith"}],
  #[[5]] != "Baker" && #[[1]] != "Cooper" &&
    #[[1]] != "Fletcher" && #[[5]] != "Fletcher" &&
    #~p~"Miller" > #~p~"Cooper" &&
    Abs[#~p~"Smith" - #~p~"Fletcher"] > 1 &&
    Abs[#~p~"Cooper" - #~p~"Fletcher"] > 1 &]]
