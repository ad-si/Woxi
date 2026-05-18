matchFrom[url_String, str_String] := Select[StringSplit[Import[url, "String"], "\n"], StringMatchQ[str]]

getLogLinks[n_] := 
 Select[Import["http://tclers.tk/conferences/tcl/", "Hyperlinks"], 
     First[
     StringCases[#1, "tcl/" ~~ date__ ~~ ".tcl" :> DateDifference[DateObject[URLDecode[date], TimeZone -> "Europe/Berlin"], Now]]] <= 
         Quantity[n, "Days"] & ]

searchLogs[str_String] := Block[{data},
  Map[
   (data = matchFrom[#, str];
     If[Length[data] > 0, 
      Print /@ Join[{#, "-----"}, data, {"----\n"}]]) &, 
   getLogLinks[10]]]

searchLogs["*lazy*"];
