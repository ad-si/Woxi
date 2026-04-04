DisplayForm[FractionBox["x", "y"]]
DisplayForm[SqrtBox["x"]]
DisplayForm[SuperscriptBox["x", "2"]]
DisplayForm[SubscriptBox["a", "i"]]
DisplayForm[RowBox[{"a", "+", "b"}]]
DisplayForm[OverscriptBox["x", "^"]]
DisplayForm[UnderscriptBox["x", "_"]]
DisplayForm[SubsuperscriptBox["x", "i", "2"]]
DisplayForm[UnderoverscriptBox["\[Sum]", "i=1", "n"]]
DisplayForm[RadicalBox["x", "3"]]
DisplayForm[FrameBox["hello"]]
DisplayForm[StyleBox["red text", FontColor -> Red]]
DisplayForm[GridBox[{{"a", "b"}, {"c", "d"}}]]
DisplayForm[StyleBox["big", FontSize -> 24]]
DisplayForm[RowBox[{FractionBox["1", "2"], "+", SqrtBox["3"]}]]
DisplayForm[TagBox[FrameBox["tagged"], "MyTag"]]
DisplayForm[StyleBox[FrameBox["styled box"], Background -> LightBlue]]
DisplayForm[RowBox[
  {SuperscriptBox["e", RowBox[{"i", " ", "\[Pi]"}]], "+", "1"}
]]
DisplayForm[GridBox[
  {{"x", SuperscriptBox["x", "2"]}, {"y", SuperscriptBox["y", "2"]}},
  GridLines -> Automatic
]]
DisplayForm[InterpretationBox["display", 42]]
