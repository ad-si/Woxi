---
icon: lucide/sigma
---

# Rubi

[Rubi] — the **Ru**le-**b**ased **I**ntegrator — is a Wolfram Language package
that computes indefinite integrals by applying a decision tree of about 7000
rewrite rules. It is not part of Woxi: it is an ordinary `.m` package that Woxi
reads and runs, and one of the largest Wolfram Language codebases there is, so
it doubles as a stress test for Woxi's pattern matcher and definition store.

[Rubi]: https://rulebasedintegration.org


## Installing

A [release] ships as a `.paclet` file, which is a ZIP archive:

```sh
curl -L -O \
  https://github.com/RuleBasedIntegration/Rubi/releases/download/4.17.3.0/Rubi-4.17.3.0.paclet
unzip Rubi-4.17.3.0.paclet
```

That unpacks a `Rubi-4.17.3.0/` directory holding `Rubi.m`, the
`IntegrationRules/` tree and a `PacletInfo.m`.
No patching is needed — Woxi reads the package as it ships.

[release]: https://github.com/RuleBasedIntegration/Rubi/releases


## Loading

Either read `Rubi.m` directly:

```wolfram
$LoadShowSteps = False;
Get["Rubi-4.17.3.0/Rubi.m"];
```

or point `PacletDirectoryLoad` at the directory the paclet was unpacked *into*
and load it by context:

```wolfram
PacletDirectoryLoad["."];
$LoadShowSteps = False;
Needs["Rubi`"];
```

Both read the same 200 rule files and end with about 7400 `Int` down-values.
`$RubiVersion` reports which version is loaded.

Loading takes roughly a minute and about a gigabyte of memory, and it happens
again in every process — the `.mx` fast-load path in the paclet's
`Kernel/init.m` needs `DumpSave`, which Woxi does not implement, so nothing is
cached between runs.

`$LoadShowSteps = False` switches off Rubi's step-display machinery. Left at
its default of `True`, Rubi additionally rewrites all 7400 rules to record
the steps they take, which costs about two more minutes of loading and makes
`Steps`, `Step` and `Stats` available (see below). `Int` itself is the same
either way.


## Integrating

`Int[expr, x]` is the antiderivative of `expr` with respect to `x`:

```wolfram
Int[1/(a + b*x), x]
(* Log[a + b*x]/b *)

Int[x^2*Sqrt[a^2 - x^2], x]
(* (Sqrt[a^2 - x^2]*x^3)/4 - (a^2*x*Sqrt[a^2 - x^2])/8
     + (a^4*ArcTan[x/Sqrt[a^2 - x^2]])/8 *)
```

A list of integrands is integrated term by term:

```wolfram
Int[{Sin[x], x*Log[x]}, x]
(* {-Cos[x], -1/4*x^2 + (x^2*Log[x])/2} *)
```

and an iterator gives the difference of the antiderivative's limits at the
endpoints:

```wolfram
Int[x^2, {x, 0, 1}]
(* 1/3 *)
```

Rubi's answers are usually not the ones `Integrate` gives — that is the point
of the package. Woxi's own `Integrate` is unaffected by loading Rubi; the two
live side by side.


## Showing the steps

With the step display loaded (`$LoadShowSteps` left at `True`),
`Steps[Int[expr, x]]` prints the rule applied at every step; with
`RubiPrintInformation -> False` it instead returns the recorded steps and the
result. Each intermediate expression arrives as a
`RubiIntermediateResult[HoldComplete[…]]`, the integrals still to be done
written as `Int[…]`:

```wolfram
{steps, result} = Steps[Int[(x^2 + x + 1)/(x^4 + x^3 + x + 1), x],
  RubiPrintInformation -> False];
Cases[steps, RubiIntermediateResult[HoldComplete[e_]] :> HoldForm[e], Infinity]
(* {Int[1/(3*(1 + x)^2) + 2/(3*(1 - x + x^2)), x],
    -1/(3*(1 + x)) + 2/3 ⋆ Int[1/(1 - x + x^2), x],
    -1/(3*(1 + x)) - 4/3 ⋆ Subst[Int[1/(-3 - x^2), x], x, -1 + 2*x],
    -1/(3*(1 + x)) - (4*ArcTan[(1 - 2*x)/Sqrt[3]])/(3*Sqrt[3])} *)
```

The `⋆` is Rubi's `Star`, a coefficient kept in front of an integral for
display. Rubi also defines how `Int` is typeset, so the steps convert to TeX
as integrals — `Convert`TeX`ExpressionToTeX[HoldForm[e]]` gives
`\int \frac{1}{3 (1+x)^2}+\frac{2}{3 \left(1-x+x^2\right)} \, dx` for the
first one — which is what the *IntWithStepsOfTeXForm* notebook builds its
`aligned` environment from.


## What does not work yet

Rubi loads unmodified and integrates, but it is not fully supported. On a
30-integral sample, 20 answers are character-for-character what
`wolframscript` gets from the same package, and 6 of the remaining 10 are the
same function written another way. What is left:

- **Loading is slow.** About a minute against roughly twenty seconds under
  `wolframscript` (three minutes with the step display), and there is no
  `.mx` cache to make the second run faster. Rubi's progress bar is a
  `Monitor`, which Woxi does not implement, so the load also reports that.
- **Powers times a trigonometric function fall through.**
  `Int[Sin[x]/x^3, x]`, `Int[x^2*Sin[x], x]` and `Int[Cos[x]/x^2, x]` come
  back as `Int[…]`: the `(c + d x)^m Sin[e + f x]` rules never fire, so a
  step display of them shows nothing either.
- **Two integrals exhaust memory.** `Int[Sin[x]^3*Cos[x]^2, x]` and
  `Int[Sin[x]*Cos[x]^3, x]` run for minutes and are killed;
  `wolframscript` answers both instantly.
- **Some step sequences take another road.** `Int[Sqrt[x + Sqrt[x]], x]`
  ends in a hypergeometric function where Rubi reaches an inverse
  hyperbolic tangent, and the terms of an intermediate sum can come out in
  a different order — both are the rule-ordering differences above seen
  step by step. The rule text recorded next to each step keeps its `FreeQ`
  conditions and writes them in linear box syntax rather than `DisplayForm`.
- **`Int[ArcSin[x], x]` comes back unevaluated.** The rule that should fire
  sits behind the `Unintegrable` fallback meant to catch what it declines —
  Woxi ranks two rules by how much structure each pattern carries, where the
  language asks whether one's match set is inside the other's.
- **Answers can be shaped differently.** `Int[Sec[x]^2, x]` is `Sec[x]*Sin[x]`
  rather than `Tan[x]`, `Int[E^x*x, x]` is `-Gamma[2, -x]` rather than
  `E^x*(x - 1)`, and sums come out in Woxi's own `Plus` order. Same functions,
  reached along a different path.
- **`Int[x/(a + b*x^2), x]` is off by a constant**: `Log[1 + b x^2/a]/(2 b)`
  where Rubi gives `Log[a + b x^2]/(2 b)`.

The current divergences are catalogued in
[Conformance gaps](comparison/mathematica/conformance_gaps.md).
