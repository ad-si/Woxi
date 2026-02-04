#!/usr/bin/env wolframscript -print -format PNG

ListLinePlot[
  RandomFunction[
    WienerProcess[],
    {0,10,0.01},
    10
  ]
]
