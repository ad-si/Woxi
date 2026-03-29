(* Data Analysis *)
data = Import[FileNameJoin[{DirectoryName[$InputFileName], "data.csv"}]];
rows = Rest[data];

(* Overall fruit consumption per person *)
byPerson = GroupBy[rows, #[[2]] &];
perPerson = (Total[#[[All, 4]]] &) /@ byPerson;
chart1 = BarChart[
  Values[perPerson],
  ChartLabels -> (Rotate[#, -Pi/4] & /@ Keys[perPerson]),
  PlotLabel -> "Per Person"
];

(* Overall fruit consumption per month *)
byMonth = GroupBy[
  rows,
  DateString[#[[1]],
  {"Year", "-","Month"}] &
];
perMonth = (Total[#[[All, 4]]] &) /@ byMonth;
chart2 = BarChart[
  Values[perMonth],
  ChartLabels -> (Rotate[#, -Pi/4] & /@ Keys[perMonth]),
  PlotLabel -> "Per Month"
];

(* Most popular fruits *)
byFruit = GroupBy[rows, #[[3]] &];
perFruit = (Total[#[[All, 4]]] &) /@ byFruit;
chart3 = BarChart[
  Values[perFruit],
  ChartLabels -> (Rotate[#, -Pi/4] & /@ Keys[perFruit]),
  PlotLabel -> "Per Fruit"
];

(* Combine all charts and export as SVG *)
chartPath = FileNameJoin[{DirectoryName[$InputFileName], "chart.svg"}];
Export[chartPath, GraphicsGrid[{{chart1, chart2, chart3}}, ImageSize -> 720], "SVG"];
Print["Chart saved to: ", chartPath];
