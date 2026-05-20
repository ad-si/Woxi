Options[OptionalSort]={ordering->lexicographic,column->1,reverse-> False};
Print[OptionalSort[x_List,OptionsPattern[]]:=If[OptionValue[reverse]==True,
SortBy[x ,#[[OptionValue[column]]]&]//Reverse,
SortBy[x,#[[OptionValue[column]]]&] ]];
Print[OptionalSort[{{"a" ,"b", "c"}, {"", "q", "z"},{"zap" ,"zip", "Zot"}} ]];
Print[OptionalSort[{{"a" ,"b", "c"}, {"", "q", "z"},{"zap" ,"zip", "Zot"}},{ordering->lexicographic,column->2,reverse-> True} ]];
