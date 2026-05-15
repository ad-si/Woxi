ExportString[
  Graphics[
    {
      (* Glass envelope *)
      LightGray, Opacity[0.15],
      Rectangle[{-1, -4.5}, {18, 4.5}],
      Opacity[1],
      Gray, Thickness[0.003],
      Line[{{-1, -4.5}, {18, -4.5}, {18, 4.5}, {-1, 4.5}, {-1, -4.5}}],

      (* Heater / Hot cathode *)
      Orange, Thickness[0.005],
      Line[{{0, -1.2}, {0.2, -0.8}, {-0.2, -0.4}, {0.2, 0}, {-0.2, 0.4}, {0.2, 0.8}, {0, 1.2}}],

      (* Cathode (K) - surface *)
      Gray, Thickness[0.005],
      Line[{{0.8, -1.5}, {0.8, 1.5}}],

      (* Wehnelt cylinder *)
      Darker[Gray], Thickness[0.004],
      Line[{{1.5, -2.0}, {1.5, -0.6}}],
      Line[{{1.5, 0.6}, {1.5, 2.0}}],

      (* Anode (aperture) *)
      Darker[Gray], Thickness[0.004],
      Line[{{3.5, -2.0}, {3.5, -0.4}}],
      Line[{{3.5, 0.4}, {3.5, 2.0}}],

      (* Electron beam - horizontal up to middle of plates *)
      Blue, Thickness[0.003], Dashing[{0.1, 0.05}],
      Arrow[{{0.8, 0}, {7.5, 0}}],

      (* Electron beam - deflected (parabolic upward) *)
      Blue, Thickness[0.003], Dashing[{0.1, 0.05}],
      BezierCurve[{{7.5, 0}, {9, 0}, {10.5, 0.8}, {11.5, 1.5}}],

      (* Continuing to the screen *)
      Blue, Dashing[{}], Thickness[0.003],
      Arrow[{{11.5, 1.5}, {16.5, 3.2}}],

      (* Deflection plates *)
      Darker[Red], Thickness[0.006], Dashing[{}],
      Line[{{6.5, 2.2}, {10.5, 2.2}}],  (* upper plate + *)
      Line[{{6.5, -2.2}, {10.5, -2.2}}], (* lower plate - *)

      (* Fluorescent screen *)
      Darker[Green], Thickness[0.006],
      Line[{{17, -4.2}, {17, 4.2}}],

      (* Bright spot *)
      Yellow, PointSize[0.02],
      Point[{17, 3.2}],

      (* E-field arrows between the plates *)
      Red, Thickness[0.002], Dashing[{}],
      Arrow[{{8.5, 1.8}, {8.5, -1.8}}],
      Arrow[{{7.5, 1.8}, {7.5, -1.8}}],
      Arrow[{{9.5, 1.8}, {9.5, -1.8}}],

      (* Plus/Minus on the plates *)
      Darker[Red], Text[Style["+", 18, Bold], {10.9, 2.2}],
      Darker[Red], Text[Style["−", 18, Bold], {10.9, -2.2}],

      (* Labels *)
      Black,
      Text[Style["Heater", 11, Italic], {0, -2.5}],
      Text[Style["Cathode (K)", 11, Italic], {0.8, -2.5}],
      Text[Style["Wehnelt\ncylinder", 11, Italic, TextAlignment -> Center], {1.5, 3.2}],
      Text[Style["Anode (A)", 11, Italic], {3.5, -3.0}],
      Text[Style["Deflection plates", 12, Bold], {8.5, 3.5}],
      Text[Style["Fluorescent screen", 11, Italic], {17, -5.0}],
      Text[Style["Electron beam", 11, Italic, Blue], {5, 0.7}],

      (* E-field label *)
      Red, Text[Style["E", 13, Bold, Italic], {9.0, 0}],

      (* Length d of the plates *)
      Black, Thickness[0.001],
      Arrow[{{6.5, -3.2}, {10.5, -3.2}}],
      Arrow[{{10.5, -3.2}, {6.5, -3.2}}],
      Text[Style["d (plate length)", 10, Italic], {8.5, -3.7}],

      (* Distance between the plates *)
      Thickness[0.001],
      Arrow[{{11.5, 2.2}, {11.5, -2.2}}],
      Arrow[{{11.5, -2.2}, {11.5, 2.2}}],
      Text[Style["Plate\nspacing", 10, Italic, TextAlignment -> Center], {12.8, 0}],

      (* === Electrical wiring === *)

      (* Heater voltage UH *)
      Darker[Orange], Thickness[0.002], Dashing[{}],
      Line[{{0, -1.2}, {0, -5.5}, {0.8, -5.5}, {0.8, -1.5}}],
      (* Battery symbol UH *)
      Thickness[0.004],
      Line[{{0.25, -5.5}, {0.25, -5.0}}],  (* short stroke - *)
      Line[{{0.55, -5.5}, {0.55, -4.8}}],  (* long stroke + *)
      Text[Style["U_H", 11, Bold, Darker[Orange]], {0.4, -6.1}],

      (* Accelerating voltage UB: from cathode to anode *)
      Darker[Blue], Thickness[0.002],
      Line[{{0.8, -1.5}, {0.8, -7.5}, {3.5, -7.5}, {3.5, -2.0}}],
      (* Battery symbol UB *)
      Thickness[0.004],
      Line[{{1.9, -7.5}, {1.9, -7.0}}],
      Line[{{2.3, -7.5}, {2.3, -6.8}}],
      Text[Style["U_B", 12, Bold, Darker[Blue]], {2.1, -8.2}],
      Text[Style["−", 14, Bold, Darker[Blue]], {1.5, -7.5}],
      Text[Style["+", 14, Bold, Darker[Blue]], {2.7, -7.5}],

      (* Deflection voltage UA: from lower to upper plate *)
      Darker[Red], Thickness[0.002],
      Line[{{6.5, 2.2}, {5.5, 2.2}, {5.5, -7.5}}],
      Line[{{6.5, -2.2}, {5.0, -2.2}, {5.0, -7.5}}],
      (* Battery symbol UA *)
      Thickness[0.004],
      Line[{{5.1, -7.5}, {5.1, -7.0}}],
      Line[{{5.4, -7.5}, {5.4, -6.8}}],
      Text[Style["U_A (deflection voltage)", 11, Bold, Darker[Red]], {5.25, -8.3}],
      Text[Style["−", 14, Bold, Darker[Red]], {4.75, -7.5}],
      Text[Style["+", 14, Bold, Darker[Red]], {5.75, -7.5}],

      (* v0 arrow at the anode exit *)
      Darker[Green], Thickness[0.003],
      Arrow[{{3.8, 0.5}, {5.5, 0.5}}],
      Text[Style["v₀", 13, Bold, Darker[Green]], {4.7, 1.0}]
    },
    PlotRange -> {{-2, 19}, {-9, 5.5}},
    ImageSize -> 800,
    Background -> White,
    PlotLabel -> Style["Schematic of an electron deflection tube", 16, Bold]
  ],
  "SVG"
]
