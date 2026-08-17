(* Post-substitution affine systems extracted from tower/wl/rule54/data/*_capture.json.
   They deliberately contain only the solver boundary: package loading, JSON
   traversal, and association evaluation are tested by Tower itself. *)

delay1 = s >= 0 && s <= 25 && -4 s + 709 >= 600 &&
  -4 s + 709 >= 641 && -4 s + 711 <= -4 s + 740 &&
  -4 s + 740 <= 800 && -4 s + 742 <= 800 &&
  -8 s + 1304 > 1080;
delay2 = s >= 0 && s <= 24 && 4 s + 597 >= 600 &&
  4 s + 597 >= 641 && 4 s + 599 <= 4 s + 630 &&
  4 s + 630 <= 800 && 8 s + 1102 > 1080;
Print[Length[Solve[delay1, s, Integers]] +
  Length[Solve[delay2, s, Integers]]];

static1 = s >= 0 && s <= 62 && -8 s + 914 >= 641 &&
  -8 s + 923 <= 800 && -8 s + 925 <= 800 && -4 s + 400 >= 128;
static2 = s >= 0 && s <= 62 && -8 s + 918 >= 641 &&
  -8 s + 927 <= 800 && -8 s + 929 <= 800 && -4 s + 402 >= 128;
static3 = s >= 0 && s <= 74 && 8 s + 296 >= 641 &&
  8 s + 311 <= 800 && 8 s + 313 <= 800 && 4 s + 102 >= 128;
static4 = s >= 0 && s <= 75 && 8 s + 292 >= 641 &&
  8 s + 307 <= 800 && 8 s + 309 <= 800 && 4 s + 100 >= 128;
Print[Length[Solve[static1, s, Integers]] +
  Length[Solve[static2, s, Integers]] +
  Length[Solve[static3, s, Integers]] +
  Length[Solve[static4, s, Integers]]];

(* Each handoff branch retains its actual indirect transport equation, a
   transport-integrality congruence, and the terminal-worldline disequality. *)
handoff1 = r >= 0 && r <= 0 && g >= 0 && h >= 0 && h <= 8 && g == 0 &&
  8 h + 822 == 822 && -8 m == -64 &&
  Mod[-8 m - 8 h - 704, 8] == 0 &&
  -512 h + 512 m - 27904 != -512 m + 512 h + 56576;
handoff2 = r >= 0 && r <= 0 && g >= 0 && h >= 0 && h <= 8 && g == 0 &&
  8 h + 814 == 822 && -8 m == -64 &&
  Mod[8 h - 8 m - 376, 8] == 0 &&
  -512 h + 512 m - 27904 != -512 m + 512 h + 56576;
handoff3 = r >= 0 && r <= 0 && g >= 0 && h >= 0 && h <= 8 && g == 0 &&
  8 h + 814 == 822 && -8 m == -64 &&
  Mod[8 m - 8 h + 368, 8] == 0 &&
  -512 h + 512 m - 27392 != -512 m + 512 h + 56064;
handoff4 = r >= 0 && r <= 0 && g >= 0 && h >= 0 && h <= 8 && g == 0 &&
  8 h + 806 == 822 && -8 m == -64 &&
  Mod[8 h + 8 m + 712, 8] == 0 &&
  -512 h + 512 m - 27392 != -512 m + 512 h + 56064;
Print[Length[Solve[handoff1, {r, g, h, m}, Integers]] +
  Length[Solve[handoff2, {r, g, h, m}, Integers]] +
  Length[Solve[handoff3, {r, g, h, m}, Integers]] +
  Length[Solve[handoff4, {r, g, h, m}, Integers]]];

derived = Reduce[-4 s + 709 >= 641 && -4 s + 740 <= 800 &&
  -4 s + 742 <= 800 && -8 s + 1304 > 1080, s, Integers];
Print[{Min[s /. Solve[derived && -100000 <= s <= 100000, s, Integers]],
  Max[s /. Solve[derived && -100000 <= s <= 100000, s, Integers]]}];
