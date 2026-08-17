---
icon: lucide/diff
---

# Wolfram Language conformance gaps

Known places where Woxi's output differs from `wolframscript`'s.
Each entry was found by diffing the two engines and verified against the
Wolfram Language, but is not fixed yet.
Roughly in descending order of user impact.

## InputForm does not normalize function spellings to operator syntax

`ToString[…, InputForm]` (and plain output) prints the function-call spelling
where WL prints the operator, so the two spellings of the same expression
render differently:

```sh
# In a result (an unevaluated ReplaceAll is what a failed replacement returns):
wolframscript -code 'ToString[ReplaceAll[x, 5], InputForm]'         # x /. 5
woxi eval 'ToString[ReplaceAll[x, 5], InputForm]'                   # ReplaceAll[x, 5]

# In a held expression:
wolframscript -code 'ToString[Unevaluated[Map[f, g]], InputForm]'   # f /@ g
woxi eval 'ToString[Unevaluated[Map[f, g]], InputForm]'             # Map[f, g]
```

The affected heads are the ones with operator spellings: `ReplaceAll` (`/.`),
`ReplaceRepeated` (`//.`), `Map` (`/@`), `Apply` (`@@`), and `Part` (`[[…]]`).
`Part` is the mildest of these — it renders correctly on its own and only
diverges inside a hold (`ToString[Unevaluated[Part[a, 2]], InputForm]`).

## Unary minus inside a pure function round-trips as a subtraction

```sh
wolframscript -code 'ToString[Unevaluated[-# &], InputForm]'  # -#1 &
woxi eval 'ToString[Unevaluated[-# &], InputForm]'            # 0 - #1 &
```

Same family as the known `Not[#2]` vs `!#2` rendering gap.

## Association inside Unevaluated echoes with the wrong spelling

```sh
wolframscript -code 'ToString[Unevaluated[<|1 -> a|>], InputForm]'  # Association[1 -> a]
woxi eval 'ToString[Unevaluated[<|1 -> a|>], InputForm]'            # <|1 -> a|>
```

Cosmetic, and confined to held echoes: `ToString[<|1 -> a|>, InputForm]` agrees
on both engines.

## ConvexHullMesh is unevaluated for 3D point sets

```sh
wolframscript -code 'ToString[ConvexHullMesh[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}], InputForm]'
# BoundaryMeshRegion[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
#   {Polygon[{{3, 2, 1}, {2, 4, 1}, {4, 3, 1}, {3, 4, 2}}]},
#   Method -> {"SeparateBoundaries" -> False}, WorkingPrecision -> Infinity]
woxi eval 'ConvexHullMesh[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}]'
# ConvexHullMesh[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]
```

Computing the hull is the easy part; WL hands qhull's internal facet
bookkeeping straight through, and three things would have to be replicated to
match the printed `Polygon`:

1. **Facet order.** No sort explains all three samples below.
2. **Vertex rotation within a face.** Faces are outward-oriented, but the
   starting vertex varies.
3. **Coplanar merging.** Triangles that share a plane come back as one polygon,
   so the cube's six faces are quads, not twelve triangles.

Reference outputs (all carry `Method -> {"SeparateBoundaries" -> False},
WorkingPrecision -> Infinity`):

| points | faces |
| --- | --- |
| `{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}` | `{{3,2,1},{2,4,1},{4,3,1},{3,4,2}}` |
| the same plus `{1,1,1}` | `{{3,2,1},{2,4,1},{4,3,1},{3,5,2},{5,4,2},{4,5,3}}` |
| the eight unit-cube corners | `{{3,2,1,4},{1,2,6,5},{4,1,5,8},{2,3,7,6},{3,4,8,7},{5,6,7,8}}` |

## `Integrate[Log[Sin[x]], …]` is unimplemented

```sh
wolframscript -code 'ToString[Integrate[Log[Sin[x]], {x, 0, 1}], InputForm]'
# (-1/12*I)*(6 + (-6 + Pi)*Pi - (12*I)*Log[2] - 6*PolyLog[2, E^(2*I)])
woxi eval 'Integrate[Log[Sin[x]], {x, 0, 1}]'
# Integrate[Log[Sin[x]], {x, 0, 1}]
```

The antiderivatives are known:

```wolfram
Integrate[Log[Sin[x]], x] ==
  -(x*Log[1 - E^((2*I)*x)]) + x*Log[Sin[x]] + (I/2)*(x^2 + PolyLog[2, E^((2*I)*x)])
Integrate[Log[Cos[x]], x] ==
  (I/2)*x^2 - x*Log[1 + E^((2*I)*x)] + x*Log[Cos[x]] + (I/2)*PolyLog[2, -E^((2*I)*x)]
```

The limit at `x -> 0` contributes `I Pi^2/12`. The blocker is not the value but
the printed form: WL's answer is `Simplify`-collected, and Woxi does not land
on that grouping.

## Inexact zero loses WL's accuracy tracking

```sh
wolframscript -code 'ToString[PascalBinomial[6.0, -2], InputForm]'  # 0``15.954589770191005
woxi eval 'PascalBinomial[6.0, -2]'                                 # 0.
```

`Binomial[6.0, -2]` has the same reference value but drifts further — Woxi
returns an exact `0` there, losing the inexactness too. WL returns an
arbitrary-precision zero whose
*accuracy* is `$MachinePrecision` — not a machine `0.`, whose accuracy is 323.6.
Reproducing it needs precision/accuracy tracking through the Gamma-ratio path
starting from a machine-real argument.

## Plus orders a reciprocal monomial before a power of a sum

```sh
wolframscript -code 'ToString[1/x + Sqrt[1 - x^(-2)], InputForm]'  # Sqrt[1 - x^(-2)] + x^(-1)
woxi eval 'ToString[1/x + Sqrt[1 - x^(-2)], InputForm]'            # x^(-1) + Sqrt[1 - x^(-2)]
```

Only the *reciprocal* monomial diverges — `x + Sqrt[1 - x^2]` agrees, and the
`Times` counterpart of the rule is implemented.

## Unimplemented argument forms found by diffing against `SyntaxInformation`

Woxi's arity table can be diffed mechanically against wolframscript's own
declared signatures:

```wolfram
(* for each implemented function name *)
"ArgumentsPattern" /. SyntaxInformation[Symbol["System`" <> name]]
```

Counting the required and optional slots (skipping anything with
`OptionsPattern` or a `BlankSequence`, which are unbounded) and comparing
against Woxi's declared maximum turns up every documented argument form that
is missing. The sweep originally found 80; the 67 below are what is left.

Each one is a form wolframscript accepts and Woxi rejects on arity — so the
failure mode is an `::argt` / `::argb` / `::argx` message rather than a wrong
answer, which makes them safe but individually invisible.

| Function | Woxi | WL up to | `ArgumentsPattern` |
| --- | --- | --- | --- |
| `AbsoluteCorrelation` | 1–2 | 3 | `{_, _., _.}` |
| `AiryAiZero` | 1–1 | 2 | `{_, _.}` |
| `AiryBiZero` | 1–1 | 2 | `{_, _.}` |
| `AngerJ` | 2–2 | 3 | `{_, _, _.}` |
| `AngleBisector` | 1–1 | 2 | `{{_, _, _}, _.}` |
| `ArcCurvature` | 2–2 | 3 | `{{__}, _, _.}` |
| `BesselYZero` | 2–2 | 3 | `{_, _, _.}` |
| `BooleanMaxterms` | 2–2 | 3 | `{_, _., _.}` |
| `BooleanMinterms` | 2–2 | 3 | `{_, _., _.}` |
| `CellularAutomaton` | 1–3 | 4 | `{_, _., _., _.}` |
| `CenterArray` | 1–3 | 4 | `{_, _., _., _.}` |
| `CharacterName` | 1–1 | 2 | `{_, _.}` |
| `CircleThrough` | 1–1 | 3 | `{{__}, _., _.}` |
| `ConstantArray` | 2–2 | 3 | `{_, _, _.}` |
| `CoordinateBoundingBoxArray` | 1–3 | 4 | `{{_, _}, _., _., _.}` |
| `CoordinateBoundsArray` | 1–3 | 4 | `{{__}, _., _., _.}` |
| `Correlation` | 1–2 | 3 | `{_, _., _.}` |
| `Counts` | 1–1 | 2 | `{_, _.}` |
| `Covariance` | 1–2 | 3 | `{_, _., _.}` |
| `DigitCount` | 1–3 | 4 | `{_, _., _., _.}` |
| `DigitSum` | 1–2 | 3 | `{_, _., _.}` |
| `EulerAngles` | 1–1 | 2 | `{_, Optional[{_, _, _}]}` |
| `FindLinearRecurrence` | 1–1 | 2 | `{_, _.}` |
| `FrenetSerretSystem` | 2–2 | 3 | `{{__}, _, _.}` |
| `GammaDistribution` | 2–2 | 4 | `{_, _, _., _.}` |
| `GompertzMakehamDistribution` | 2–2 | 4 | `{_, _, _., _.}` |
| `GroupOrbits` | 2–2 | 3 | `{_, Optional[{__}], _.}` |
| `GroupStabilizer` | 2–2 | 3 | `{_, {__}, _.}` |
| `Groupings` | 2–2 | 3 | `{_, _, _.}` |
| `HarmonicNumber` | 1–2 | 3 | `{_, _., _.}` |
| `Head` | 1–1 | 2 | `{_, _.}` |
| `ImageAdjust` | 1–2 | 4 | `{_, _., Optional[{_, _}], Optional[{_, _}]}` |
| `Inner` | 3–4 | 5 | `{_, _, _, _., _.}` |
| `InverseChiSquareDistribution` | 1–1 | 2 | `{_, _.}` |
| `InverseErf` | 1–1 | 2 | `{_, _.}` |
| `InverseGammaDistribution` | 2–2 | 4 | `{_, _, _., _.}` |
| `InverseGaussianDistribution` | 2–2 | 3 | `{_, _, _.}` |
| `KendallTau` | 2–2 | 3 | `{_, _., _.}` |
| `Latitude` | 1–1 | 2 | `{_., _.}` |
| `LatitudeLongitude` | 1–1 | 2 | `{_., _.}` |
| `LegendreP` | 2–3 | 4 | `{_, _, _., _.}` |
| `LegendreQ` | 2–3 | 4 | `{_, _, _., _.}` |
| `ListConvolve` | 2–6 | 7 | `{_, _, _., _., _., _., _.}` |
| `ListCorrelate` | 2–6 | 7 | `{_, _, _., _., _., _., _.}` |
| `Longitude` | 1–1 | 2 | `{_., _.}` |
| `MaximalBy` | 1–3 | 4 | `{_, _., _., _.}` |
| `MeijerG` | 3–3 | 4 | `{{{___}, {___}}, {{___}, {___}}, _, _.}` |
| `MinimalBy` | 1–3 | 4 | `{_, _., _., _.}` |
| `MultipleHarmonicNumber` | 1–2 | 3 | `{_, Optional[{__}], Optional[{__, _}]}` |
| `NotebookDirectory` | 0–0 | 1 | `{_.}` |
| `ParentDirectory` | 0–1 | 2 | `{_., _.}` |
| `PerfectNumber` | 1–1 | 2 | `{_, _.}` |
| `PolyLog` | 2–2 | 3 | `{_, _, _.}` |
| `PositionLargest` | 1–2 | 3 | `{_, _., _.}` |
| `PositionSmallest` | 1–2 | 3 | `{_, _., _.}` |
| `Precedence` | 1–1 | 2 | `{_, _.}` |
| `QuantityQ` | 1–1 | 2 | `{_, _.}` |
| `RiceDistribution` | 2–2 | 3 | `{_, _, _.}` |
| `SpearmanRho` | 2–2 | 3 | `{_, _., _.}` |
| `Subsequences` | 1–2 | 3 | `{_, _., _.}` |
| `SubsetReplace` | 1–2 | 3 | `{_, _., _.}` |
| `Symmetrize` | 1–1 | 2 | `{_, _.}` |
| `SyntaxQ` | 1–1 | 2 | `{_, _.}` |
| `ToBoxes` | 1–1 | 2 | `{_, _.}` |
| `Uncompress` | 1–1 | 2 | `{_, _.}` |
| `Unique` | 0–1 | 2 | `{_., Optional[{__}]}` |
| `WeberE` | 2–2 | 3 | `{_, _, _.}` |

Notes on the ones already looked at:

- `MinimalBy` / `MaximalBy` / `DigitCount`: wolframscript appears to *ignore*
  the extra trailing argument (`MinimalBy[{{1,2},{3,1},{2,5}}, Last, 2, x]`
  gives the same answer as without `x`), so there may be nothing to match.
- `Counts[list, n]`: wolframscript rejects a non-list second argument with
  `Counts::invl`, so the slot is not a useful form either.
- `PositionLargest` / `PositionSmallest`: the remaining slot is an
  *orderfun*, but wolframscript rejects `Greater` and `Less` there with
  `::nord3` (it wants an `Order`-style comparison returning ±1/0), and
  `Order` itself is the default — so there is little behaviour to match.
- The generalized distribution constructors (`GammaDistribution` 2→4,
  `InverseGammaDistribution` 2→4, `GompertzMakehamDistribution` 2→4,
  `RiceDistribution` 2→3, `InverseGaussianDistribution` 2→3,
  `InverseChiSquareDistribution` 1→2) have real closed forms, but their
  output is `Piecewise` wrapping `GammaRegularized`, `LaguerreL` and nested
  `Gamma` quotients. Matching wolframscript there means matching its exact
  symbolic canonicalization.

```sh
wolframscript -code 'ToString[PDF[GammaDistribution[a, b, g, m], x], InputForm]'
# Piecewise[{{(g*((-m + x)/b)^(-1 + a*g))/(b*E^((-m + x)/b)^g*Gamma[a]), x > m}}, 0]
woxi eval 'PDF[GammaDistribution[a, b, g, m], x]'
# GammaDistribution::argrx: GammaDistribution called with 4 arguments; 2 arguments are expected.
```

## ListCorrelate / ListConvolve have no multi-dimensional overhang

```sh
wolframscript -code 'ToString[ListCorrelate[{{1, 1}, {1, 1}}, {{a, b, c}, {d, e, f}, {g, h, i}}, 1], InputForm]'
# {{a + b + d + e, b + c + e + f, a + c + d + f},
#  {d + e + g + h, e + f + h + i, d + f + g + i},
#  {a + b + g + h, b + c + h + i, a + c + g + i}}
woxi eval 'ListCorrelate[{{1, 1}, {1, 1}}, {{a, b, c}, {d, e, f}, {g, h, i}}, 1]'
# ListCorrelate[{{1, 1}, {1, 1}}, {{a, b, c}, {d, e, f}, {g, h, i}}, 1]
```

The two-argument multi-dimensional form is correct; only the overhang path
(`k` / `{kL, kR}`, padding, generalized `g`/`h`) is one-dimensional, and it
stays unevaluated for a rank-2 kernel rather than answering.

The 7th argument, a level specification, is unimplemented for every rank:

```sh
wolframscript -code 'ToString[ListCorrelate[{x, y}, {a, b, c}, 1, p, Times, Plus, 1], InputForm]'
# {a*x + b*y, b*x + c*y, c*x + p*y}
woxi eval 'ListCorrelate[{x, y}, {a, b, c}, 1, p, Times, Plus, 1]'
# ListCorrelate::argb: called with 7 arguments; between 2 and 6 arguments are expected.
```

## A non-terminating NestWhile returns a wrong answer instead of not terminating

```sh
wolframscript -code 'TimeConstrained[NestWhile[#/2 &, 16, UnsameQ, 2], 5, timeout]'   # timeout
woxi eval 'NestWhile[#/2 &, 16, UnsameQ, 2]'                                          # 1/2^10000
```

Successive halvings of 16 are never `SameQ`, so this iterates forever in
wolframscript. Woxi silently stops after about 10 000 iterations and returns
the value it had reached, which is neither wolframscript's behaviour nor an
error.

## Format upvalues are stored as UpValues rather than FormatValues

```sh
wolframscript -code 'c /: Format[c] := "see"; ToString[UpValues[c], InputForm]'  # {}
woxi eval 'ClearAll[c]; c /: Format[c] := "see"; UpValues[c]'
# {HoldPattern[Format[c]] :> see}
```

Wolfram files a `Format` definition made through `/:` under `FormatValues`,
not `UpValues`, so `UpValues[c]` comes back empty. Woxi has no separate
`FormatValues` table and reports the rule as an upvalue. The rest of the
`UpValues` / `TagSet` / `TagUnset` family agrees with wolframscript.

## Outer does not report mismatched heads

```sh
wolframscript -code 'Outer[f, h[1, 2], {a, b}]'
# Outer::heads: Heads List and h at positions 3 and 2 are expected to be the same.
# Outer[f, h[1, 2], {a, b}]
woxi eval 'Outer[f, h[1, 2], {a, b}]'
# Outer[f, h[1, 2], {a, b}]
```

The result is right — the call stays unevaluated either way — but the
`Outer::heads` message is missing. Every other `Outer` form checked
(per-list levels, general heads, operator form) agrees.

## `Format[…]` is neither held nor form-aware

```sh
wolframscript -code 'Head[Format[x + y]]'          # Format
woxi eval 'Head[Format[x + y]]'                    # Plus

wolframscript -code 'Head[Format[x + y, OutputForm]]'  # Format
woxi eval 'Head[Format[x + y, OutputForm]]'            # Symbol
```

Wolfram keeps `Format[expr]` and `Format[expr, form]` unevaluated — `Length`
is 1 or 2, part 1 is the expression — and only applies a display rule, the
same model `Definition` follows.

The second argument is also ignored, so every form but `OutputForm` renders
the plain expression:

```sh
wolframscript -code 'Format[Sqrt[x], TeXForm]'      # \sqrt{x}
woxi eval 'Format[Sqrt[x], TeXForm]'                # Sqrt[x]

wolframscript -code 'Format[x/y, TeXForm]'          # \frac{x}{y}
woxi eval 'Format[x/y, TeXForm]'                    # x/y

wolframscript -code 'Format[x + y, StandardForm]'   # RowBox[{x, +, y}]
woxi eval 'Format[x + y, StandardForm]'             # x + y

wolframscript -code 'Format["ab", InputForm]'       # "ab"
woxi eval 'Format["ab", InputForm]'                 # ab
```

`StandardForm` and `TraditionalForm` want boxes, which `ToBoxes` already
produces correctly, and `TeXForm` / `InputForm` want the corresponding
renderer. An unsupported form (`Format[x^2, FullForm]`) stays unevaluated in
WL and prints as `Format[x^2, FullForm]`.

## TeXForm orders the terms of a sum by monomial degree

```sh
wolframscript -code 'ToString[TeXForm[1 + x + x^2]]'      # x^2+x+1
woxi eval 'ToString[TeXForm[1 + x + x^2]]'                # x+x^2+1

wolframscript -code 'ToString[TeXForm[a x^2 + b x + c]]'  # a x^2+b x+c
woxi eval 'ToString[TeXForm[a x^2 + b x + c]]'            # c+b x+a x^2

wolframscript -code 'ToString[TeXForm[Sin[x] + Cos[y]]]'  # \sin (x)+\cos (y)
woxi eval 'ToString[TeXForm[Sin[x] + Cos[y]]]'            # \cos (y)+\sin (x)
```

Woxi keeps the canonical order and only moves numbers (and complex atoms) to
the end; Wolfram sorts by the monomial, highest degree first. Neither
"reverse the canonical order" nor "stable sort by total degree" reproduces
all the samples — `3 + a + b` stays `a+b+3` (not reversed) while
`Cos[y] + Sin[x]` does reverse, and `x^3 + x^2 y^2` keeps `x^3` first even
though its total degree is lower — so this needs WL's actual lexicographic
monomial comparison. Everything else in a 211-expression TeXForm sweep
agrees.

## TeXForm stacks a rational coefficient that wolframscript factors out

```sh
wolframscript -code 'ToString[TeXForm[(3 x^2 - 1)/2]]'  # \frac{1}{2} \left(3 x^2-1\right)
woxi eval 'ToString[TeXForm[(3 x^2 - 1)/2]]'            # \frac{3 x^2-1}{2}
```

Same for `LegendreP[2, x]`, which evaluates to that expression in both
engines. A sum without a numeric term is stacked by both
(`(a + b)/2` → `\frac{a+b}{2}`, `3 (a + b)/2` → `\frac{3 (a+b)}{2}`), so the
trigger looks like a numeric term inside the numerator; the rule was not
pinned down.

`x^(1/(2 y))` is a smaller instance of the same class: WL writes
`x^{\left.\frac{1}{2}\right/y}`, Woxi `x^{\frac{1}{2 y}}`.

## Rasterize with an unknown element aborts the whole script

```sh
wolframscript -code 'Print[ToString[Rasterize[x, "Text"], InputForm]]; Print["after"]'
# Rasterize::elmntavl: "Text" is not an available element. Possible elements
# include "BoundingBox", "Data", "Graphics", "RasterSize", "Regions", and "Image".
# $Failed
# after
woxi eval 'Print[Rasterize[x, "Text"]]; Print["after"]'
# Error: Evaluation error: Rasterize: unsupported expression type
```

The hard error kills the run, so nothing after it evaluates. `Rasterize`
should emit `Rasterize::elmntavl` and return `$Failed`.

## SyntaxLength is unimplemented

```sh
wolframscript -code 'SyntaxLength["1+"]'   # 4
woxi eval 'SyntaxLength["1+"]'             # SyntaxLength[1+]  (+ "not yet implemented" warning)
```

`SyntaxLength[s]` is the length of the longest prefix of `s` that could still
begin a complete expression — 4 for `"1+"` (WL counts past the end of the
string, since more input could complete it).

## `Unevaluated[…]` is not transparent to structural functions

```sh
wolframscript -code 'Unevaluated[1 + 1] // Head'   # Plus
woxi eval 'Unevaluated[1 + 1] // Head'             # Unevaluated

wolframscript -code 'Depth[Unevaluated[1 + 1]]'    # 2
woxi eval 'Depth[Unevaluated[1 + 1]]'              # 3
```

`Length` and `AtomQ` already agree; `Head`, `Depth` and `First` still see the
wrapper.

## Solve over a system of two Abs equations gives no solutions

```sh
wolframscript -code 'Solve[{Abs[x] == 2, Abs[y] == 3}, {x, y}]'
# {{x -> -2, y -> -3}, {x -> 2, y -> -3}, {x -> -2, y -> 3}, {x -> 2, y -> 3}}
woxi eval 'Solve[{Abs[x] == 2, Abs[y] == 3}, {x, y}]'
# {}
```

One `Abs` equation is solved correctly, and a single `Abs` equation narrowed by
an inequality is too. It is the multi-variable elimination that cannot take two
of them apart and reports the system unsatisfiable, so `ToRules` turns `False`
into an empty list.

## Solve orders the negative root first in a two-variable system

```sh
wolframscript -code 'Solve[{x^2 + y^2 == 1, y == 0}, {x, y}]'
# {{x -> 1, y -> 0}, {x -> -1, y -> 0}}
woxi eval 'Solve[{x^2 + y^2 == 1, y == 0}, {x, y}]'
# {{x -> -1, y -> 0}, {x -> 1, y -> 0}}
```

The single-variable case agrees (`Solve[x^2 == 4, x]` is `{{x -> -2}, {x -> 2}}`
in both), so the system path sorts where wolframscript does not.

## Reduce does not eliminate quantifiers

```sh
wolframscript -code 'Reduce[Exists[y, x == y^2], x, Reals]'      # x >= 0
woxi eval 'Reduce[Exists[y, x == y^2], x, Reals]'                # Reduce[Exists[y, x == y^2], x]

wolframscript -code 'Reduce[ForAll[y, x + y^2 >= x], x, Reals]'  # True
woxi eval 'Reduce[ForAll[y, x + y^2 >= x], x, Reals]'            # Reduce[ForAll[y, x + y^2 >= x], x]
```

`Exists` and `ForAll` are parsed but never eliminated. Note the unevaluated
form also drops the `Reals` domain argument, which is a separate bug in the
echo path.

## Solve over the integers drops a range constraint

```sh
wolframscript -code 'Solve[Mod[x, 3] == 1 && 0 <= x < 10, x, Integers]'
# {{x -> 1}, {x -> 4}, {x -> 7}}
woxi eval 'Solve[Mod[x, 3] == 1 && 0 <= x < 10, x, Integers]'
# Solve[Mod[x, 3] == 1, x]
```

Bounded linear systems over the integers are already enumerated; a `Mod`
congruence with an explicit range is the same shape. The returned expression
having lost both the bound and the domain makes this look worse than a plain
unevaluated result.

## `Derivative[n][f][x]` is stored flat, so structural functions see three parts

```sh
wolframscript -code 'Head[Derivative[1][g][x]]'      # Derivative[1][g]
woxi eval 'Head[Derivative[1][g][x]]'                # Derivative

wolframscript -code 'Length[Derivative[1][g][x]]'    # 1
woxi eval 'Length[Derivative[1][g][x]]'              # 3

wolframscript -code 'Apply[f, Derivative[1][g][x]]'  # f[x]
woxi eval 'Apply[f, Derivative[1][g][x]]'            # f[1, g, x]
```

Woxi stores `Derivative[1][g][x]` flat rather than as nested curried calls.
The renderer prints it correctly and `D` returns the right thing, so only
structural introspection diverges — `Head`, `Length`, `Part`, `Level`, `Map`
and `Apply` all leak the internal shape.

## Positive and Sign underflow on a rational below the double range

```sh
wolframscript -code 'Positive[10^-400]'   # True
woxi eval 'Positive[10^-400]'             # False

wolframscript -code 'Sign[10^-400]'       # 1
woxi eval 'Sign[10^-400]'                 # 0
```

`Power` and the comparison operators compare such values exactly, but
`Positive`, `Negative` and `Sign` still route through `f64`, where the value
underflows to `0.0`. They should take the sign from the numerator of an exact
rational instead.

## Machine floats overflow where wolframscript promotes to arbitrary precision

```sh
wolframscript -code 'Exp[1000] // N'   # 1.97007111401704699388887935224`12.95…*^434
woxi eval 'Exp[1000] // N'             # Infinity

wolframscript -code '1.0*^308 * 10'    # 1.00000000000000001097906362944`15.95…*^309
woxi eval '1.0*^308 * 10'              # Infinity
```

A machine-real result outside the `f64` range becomes `Infinity` in Woxi,
while wolframscript widens to an arbitrary-precision number carrying its own
precision tag. Anything that grows past ~1.8*^308 is affected, so
`Exp`/`Power` of a few hundred silently loses the answer rather than
approximating it.

## An operator form that fails names the flattened call, not what was written

```sh
wolframscript -code 'Select[EvenQ][5]'
# Select::normal: Nonatomic expression expected at position 1 in Select[EvenQ][5].
# Select[EvenQ][5]
woxi eval 'Select[EvenQ][5]'
# Select::normal: Nonatomic expression expected at position 1 in Select[5, EvenQ].
# Select[5, EvenQ]
```

`f[spec][data]` is rewritten to `f[data, spec]` before dispatch, and when the
call then fails both the message and the echo describe the rewrite. The same
happens for `SortBy`, `KeyTake` and `Nearest`, so this is one issue in the
operator-form dispatch rather than a per-function one.

## Three Limit shapes at infinity stay unevaluated

```sh
wolframscript -code 'Limit[x^100/Exp[x], x -> Infinity]'        # 0
woxi eval 'Limit[x^100/Exp[x], x -> Infinity]'                  # Limit[x^100/E^x, x -> Infinity]

wolframscript -code 'Limit[Log[Log[x]]/Log[x], x -> Infinity]'  # 0
woxi eval 'Limit[Log[Log[x]]/Log[x], x -> Infinity]'            # Limit[…] unevaluated

wolframscript -code 'Limit[x/(x + Sqrt[x]), x -> -Infinity]'    # 1
woxi eval 'Limit[x/(x + Sqrt[x]), x -> -Infinity]'              # Limit[…] unevaluated
```

Each has its own cause. A polynomial over an exponential resolves up to about
degree 45 and then hits the L'Hôpital depth guard. Nested logarithms need a
`u = Log[x]` substitution — `Limit[Log[u]/u, u -> Infinity]` is already 0.
The `-Infinity` case is excluded on purpose: the leading-order analysis is
gated to `+Infinity` because `x^p` for non-integer `p` is not real to the
left of zero, and extending it needs branch-cut care.

None of these returns a wrong value, only an unevaluated one.

## Total groups negative levels globally rather than per parent

```sh
wolframscript -code 'Total[{{1, 2}, {3, {4, 5}}}, {-1}]'   # {3, {3, 9}}
woxi eval 'Total[{{1, 2}, {3, {4, 5}}}, {-1}]'             # {{1, 2}, {3, 9}}
```

Woxi reads `{-1}` as the deepest level of the whole expression — level 3 here,
so it answers `Total[…, {3}]`. wolframscript sums the depth-1 parts grouped by
their immediate parent, which leaves the depth-1 atom `3` alone and reduces
only `{4, 5}`.

Every other head measures a negative level as the part's own depth; `Total`
is the exception because its traversal also carries `AllowedHeads` and the
head-preservation rules.

## ImageAdd and friends refuse mismatched dimensions

```sh
wolframscript -code 'ImageData[ImageAdd[Image[{{0.1, 0.2}}], Image[{{0.3}}]]]'
# {{0.4, 0.2}}
wolframscript -code 'ImageData[ImageAdd[Image[{{0.3}}], Image[{{0.1, 0.2}}]]]'
# {{0.5}}
woxi eval 'ImageData[ImageAdd[Image[{{0.1, 0.2}}], Image[{{0.3}}]]]'
# Error: Evaluation error: ImageAdd: images must have the same dimensions and channels
```

The result always takes the first image's dimensions, but the second image is
read differently depending on which is larger: a smaller second image is
applied at the top-left and the rest of the first is passed through, while a
larger one contributes its *last* element (`0.3 + 0.2 = 0.5`), not its first.
`ImageMultiply` and `ImageSubtract` behave the same way.

Refusing is at least honest, but it refuses with a hard error, which aborts the
enclosing evaluation.

## Four image heads accept a bad specification silently

```sh
wolframscript -code 'ImageCrop[Image[{{0.1, 0.2}}], x]'
# ImageCrop::arg2: x is not a positive integer, pair of integers, Full or Automatic.
woxi eval 'ImageCrop[Image[{{0.1, 0.2}}], x]'
# ImageCrop[-Image-, x]     (no message)

wolframscript -code 'ImagePad[Image[{{0.1, 0.2}}], x]'
# ImagePad::imgpadn: Expecting a number or a 2 by 2 matrix of numbers instead of x.
woxi eval 'ImagePad[Image[{{0.1, 0.2}}], x]'
# ImagePad[-Image-, x]      (no message)

wolframscript -code 'TotalVariationFilter[{1, 2, 3, 4, 100}, -1]'
# TotalVariationFilter::arg2: Expecting a non-negative real number, a vector of
# such numbers (for multi-channel images) or Automatic instead of -1.
woxi eval 'TotalVariationFilter[{1, 2, 3, 4, 100}, -1]'
# TotalVariationFilter[{1, 2, 3, 4, 100}, -1]   (no message)
```

These return the right expression and only the message is missing, so they are
conformance gaps rather than wrong answers. Note that
`TotalVariationFilter`'s second argument is a regularisation parameter, not a
neighbourhood range, so it does not share the other filters' validation.

## FindThreshold is unimplemented and its return convention is unclear

```sh
wolframscript -code 'FindThreshold[Image[{{0., 1.}}]]'   # 0.498046875
woxi eval 'FindThreshold[Image[{{0., 1.}}]]'
# FindThreshold[-Image-] is a built-in Wolfram Language function not yet implemented in Woxi.
```

The Otsu machinery already exists — `EdgeDetect` uses it, and for the gradient
images it is applied to it reproduces wolframscript exactly: bin into 256 bins
spanning `[min, max]`, maximise the between-cluster variance, and return
`min + k (max - min)/256` for the winning bin index `k`.

What blocks exposing it as `FindThreshold` is that the reported value is not
always on that grid. `FindThreshold[Image[{{0., 1.}}]]` is `0.498046875`, which
is `127.5/256` — a bin *centre*, where the gradient cases return a bin *edge*.

## EdgeDetect keeps a whole flat gradient plateau

```sh
wolframscript -code 'ImageData[EdgeDetect[Image[{{0, 0, 1, 1, 0, 0, 1, 1, 0, 0}}]]]'
# {{0, 1, 0, 1, 0, 1, 0, 0, 1, 0}}
woxi eval 'ImageData[EdgeDetect[Image[{{0, 0, 1, 1, 0, 0, 1, 1, 0, 0}}]]]'
# {{0, 1, 0, 1, 1, 1, 1, 0, 1, 0}}
```

A period-4 square wave is resonant with the default radius-2 kernel and
produces a gradient magnitude that is *exactly* constant across six pixels.
wolframscript keeps alternating ones there; Woxi's non-maximum suppression
keeps the whole plateau.

No rule over the magnitudes of the two neighbours can tell index 3 from index 4
in that run — both have equal magnitudes on both sides — so the tie-break uses
information that could not be identified from the outputs. Every
non-degenerate case matches, over 15 reference cases spanning 1D and 2D shapes,
colour and Byte images, and explicit thresholds.

## LaplacianGaussianFilter and DerivativeFilter are unimplemented

```sh
wolframscript -code 'First[ImageData[LaplacianGaussianFilter[Image[{{0., 0., 0., 1., 0., 0., 0.}}], 1]]]'
# {0., 0., 1.8418419230004717, -3.683683846000945, 1.8418419230004717, 0., 0.}
wolframscript -code 'First[ImageData[DerivativeFilter[Image[{{0., 0., 0., 1., 0., 0., 0.}}], {0, 1}]]]'
# {0.05771365940052183, -0.21539030917347252, 0.803847577293368, 0., …}
```

Both are feature gaps rather than divergences. Neither can reuse
`GradientFilter` directly: `DerivativeFilter` normalises its kernel
differently (the impulse responses do not agree up to a scale factor), and
`LaplacianGaussianFilter`'s three-tap response at radius 1 is a scaled second
difference rather than the second-derivative-of-Gaussian kernel the name
suggests.

## Twenty-two heads raise a hard error for a bad first argument

```sh
wolframscript -code 'ImageCrop[{1, 2, 3}, -1]'
# ImageCrop::imgvinv: Expecting an image, graphics or video instead of {1, 2, 3}.
# ImageCrop[{1, 2, 3}, -1]
woxi eval 'ImageCrop[{1, 2, 3}, -1]'
# Error: Evaluation error: ImageCrop: first argument is not an Image
```

A sweep of the 855 heads whose arity admits two arguments, against nine
degenerate argument pairs, found 227 calls that abort the enclosing evaluation
instead of reporting. What remains is this one shape, spread over `While`,
`TuringMachine`, `Find`, `ReadList`, `Piecewise`, `ImageApply`,
`ImageAssemble`, `ImageCollage`, `ImageCompose`, `ImageCrop`, `NMinimize`,
`NMaximize`, `FindSequenceFunction`, `RealDigits`, `Dt`, `FrenetSerretSystem`,
`ArcCurvature`, `MinimalPolynomial`, `Root`, `NumberFieldSignature`,
`WordCounts` and `DivisorSigma`.

The argument is nonsense in every case, so no correct result is lost — but a
hard error takes the whole script down where a message does not.

## AssociationThread does not take a scalar key

```sh
wolframscript -code 'AssociationThread[3, {1, 2}]'      # <|3 -> 2|>
wolframscript -code 'AssociationThread["ab", {1, 2}]'   # <|ab -> 2|>
woxi eval 'AssociationThread[3, {1, 2}]'                # AssociationThread[3, {1, 2}]
```

A scalar *value* is shared across the keys, but a scalar *key* against a list
of values keeps only the last value, which is peculiar enough that
generalising from two examples would be guessing. Left unevaluated on purpose;
it echoes rather than aborting.
