---
icon: lucide/audio-waveform
---

# Wavelet Analysis

Woxi implements the Wolfram Language wavelet analysis functions:
wavelet families, filter coefficients, discrete and continuous wavelet
transforms, coefficient manipulation, and wavelet visualization.


## Wavelet Families

Wavelet families are symbolic constructor objects:

```scrut
$ wo 'HaarWavelet[]'
HaarWavelet[]
```

```scrut
$ wo 'DaubechiesWavelet[4]'
DaubechiesWavelet[4]
```


## Filter Coefficients

`WaveletFilterCoefficients` gives the filter as `{index, coefficient}`
pairs. Lowpass coefficients sum to 1:

```scrut
$ wo 'WaveletFilterCoefficients[HaarWavelet[]]'
{{0, 0.5}, {1, 0.5}}
```

```scrut
$ wo 'WaveletFilterCoefficients[HaarWavelet[], "PrimalHighpass"]'
{{0, 0.5}, {1, -0.5}}
```

Exact coefficients with `WorkingPrecision -> Infinity`:

```scrut
$ wo 'WaveletFilterCoefficients[DaubechiesWavelet[2], "PrimalLowpass", WorkingPrecision -> Infinity]'
{{0, (1 + Sqrt[3])/8}, {1, (3 + Sqrt[3])/8}, {2, (3 - Sqrt[3])/8}, {3, (1 - Sqrt[3])/8}}
```

Biorthogonal families have separate primal and dual filters:

```scrut
$ wo 'WaveletFilterCoefficients[BiorthogonalSplineWavelet[2, 2], "DualLowpass"]'
{{-1, 0.25}, {0, 0.5}, {1, 0.25}}
```


## Discrete Wavelet Transform

`DiscreteWaveletTransform` gives a `DiscreteWaveletData` object; the
coefficients are accessed with wavelet indices:

```scrut
$ wo 'dwd = DiscreteWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 1]; {Round[dwd[{0}, "Values"], 0.001], Round[dwd[{1}, "Values"], 0.001]}'
{{{2.121, 4.95}}, {{-0.707, -0.707}}}
```

```scrut
$ wo 'DiscreteWaveletTransform[{1, 2, 3, 4}]["BasisIndex"]'
{{1}, {0, 1}, {0, 0}}
```

`InverseWaveletTransform` reconstructs the data:

```scrut
$ wo 'Round[InverseWaveletTransform[DiscreteWaveletTransform[{1, 2, 3, 4}]]]'
{1, 2, 3, 4}
```

Symbolic data transforms exactly:

```scrut
$ wo 'Rationalize[Simplify[InverseWaveletTransform[DiscreteWaveletTransform[{a, b, c, d}, HaarWavelet[]]]]]'
{a, b, c, d}
```


## Lifting Wavelet Transform

The lifting transform computes exact results for exact input:

```scrut
$ wo 'Normal[LiftingWaveletTransform[{1, 1, 3, 1}, HaarWavelet[], 1, WorkingPrecision -> Infinity]]'
{{0} -> {Sqrt[2], 2*Sqrt[2]}, {1} -> {0, -Sqrt[2]}}
```


## Wavelet and Scaling Functions

```scrut
$ wo 'WaveletPhi[HaarWavelet[], x]'
Piecewise[{{1, Inequality[0, LessEqual, x, Less, 1]}}, 0]
```

The continuous Mexican-hat wavelet evaluates numerically at a point:

```scrut
$ wo 'Round[WaveletPsi[MexicanHatWavelet[1], 1/2], 0.0001]'
0.5741
```


## Thresholding

`WaveletThreshold` shrinks detail coefficients (here soft thresholding
with an explicit threshold value):

```scrut
$ wo 'WaveletThreshold[DiscreteWaveletTransform[{1., 5., 2., 8.}], {"Soft", 1.}][{1}, "Values"]'
{{-1.8284271247461903, -3.2426406871192857}}
```


## Continuous Wavelet Transform

```scrut
$ wo 'cwd = ContinuousWaveletTransform[Range[32] // N]; {cwd["Octaves"], cwd["Voices"], cwd["Wavelet"]}'
{4, 4, MexicanHatWavelet[1]}
```
