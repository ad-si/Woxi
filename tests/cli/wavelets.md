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
{{0, 1/2}, {1, 1/2}}
```

```scrut
$ wo 'WaveletFilterCoefficients[HaarWavelet[], "PrimalHighpass"]'
{{0, 1/2}, {1, -1/2}}
```

Exact coefficients with `WorkingPrecision -> Infinity`:

```scrut
$ wo 'WaveletFilterCoefficients[DaubechiesWavelet[2], "PrimalLowpass", WorkingPrecision -> Infinity]'
{{0, (1 + Sqrt[3])/8}, {1, (3 + Sqrt[3])/8}, {2, (3 - Sqrt[3])/8}, {3, (1 - Sqrt[3])/8}}
```

Biorthogonal families have separate primal and dual filters:

```scrut
$ wo 'WaveletFilterCoefficients[BiorthogonalSplineWavelet[2, 2], "DualLowpass"]'
{{-2, -1/8}, {-1, 1/4}, {0, 3/4}, {1, 1/4}, {2, -1/8}}
```


## Discrete Wavelet Transform

`DiscreteWaveletTransform` gives a `DiscreteWaveletData` object; the
coefficients are accessed with wavelet indices:

```scrut
$ wo 'dwd = DiscreteWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 1]; dwd[All]'
{{0} -> {2.121320343559643, 4.949747468305833}, {1} -> {-0.7071067811865476, -0.7071067811865476}}
```

```scrut
$ wo 'DiscreteWaveletTransform[{1, 2, 3, 4}]["BasisIndex"]'
{{0, 0}, {0, 1}, {1}}
```

`InverseWaveletTransform` reconstructs the data:

```scrut
$ wo 'Round[InverseWaveletTransform[DiscreteWaveletTransform[{1, 2, 3, 4}]]]'
{1, 2, 3, 4}
```

Symbolic data transforms exactly:

```scrut
$ wo 'Simplify[InverseWaveletTransform[DiscreteWaveletTransform[{a, b, c, d}, HaarWavelet[]]]]'
{a, b, c, d}
```


## Lifting Wavelet Transform

The lifting transform computes exact results for exact input:

```scrut
$ wo 'Normal[LiftingWaveletTransform[{1, 1, 3, 1}, HaarWavelet[], 1, WorkingPrecision -> Infinity]]'
{{0} -> {Sqrt[2], 2*Sqrt[2]}, {1} -> {0, Sqrt[2]}}
```


## Wavelet and Scaling Functions

```scrut
$ wo 'WaveletPhi[HaarWavelet[], x]'
Piecewise[{{1, Inequality[0, LessEqual, x, Less, 1]}}, 0]
```

```scrut
$ wo 'WaveletPsi[MexicanHatWavelet[1], t]'
(2*E^(-1/2*t^2)*(1 - t^2))/(Sqrt[3]*Pi^(1/4))
```


## Thresholding

`WaveletThreshold` shrinks detail coefficients (here soft thresholding
with an explicit threshold value):

```scrut
$ wo 'WaveletThreshold[DiscreteWaveletTransform[{1., 5., 2., 8.}], {"Soft", 1.}][{1}, "Values"]'
{-1.8284271247461903, -3.2426406871192857}
```


## Continuous Wavelet Transform

```scrut
$ wo 'cwd = ContinuousWaveletTransform[Range[32] // N]; {cwd["Octaves"], cwd["Voices"], cwd["Wavelet"]}'
{4, 4, MexicanHatWavelet[]}
```
