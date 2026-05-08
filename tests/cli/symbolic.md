---
icon: lucide/variable
---

# Symbolic Computing

```scrut
$ wo 'cow + 5'
5 + cow
```

```scrut
$ wo 'cow + 5 + 10'
15 + cow
```

```scrut
$ wo 'moo = cow + 5'
5 + cow
```

```scrut
$ wo 'D[x^n, x]'
n*x^(-1 + n)
```

```scrut
$ wo 'Integrate[x^2 + Sin[x], x]'
x^3/3 - Cos[x]
```


## Limits

```scrut
$ wo 'Limit[Sin[x]/x, x -> 0]'
1
```


## Series

```scrut
$ wo 'Series[Exp[x], {x, 0, 3}]'
SeriesData[x, 0, {1, 1, 1/2, 1/6}, 0, 4, 1]
```


## Apart

```scrut
$ wo 'Apart[1/(x^2 - 1)]'
1/(2*(-1 + x)) - 1/(2*(1 + x))
```


## Together

```scrut
$ wo 'Together[1/x + 1/y]'
(x + y)/(x*y)
```


## Cancel

```scrut
$ wo 'Cancel[(x^2 - 1)/(x - 1)]'
1 + x
```


## Collect

```scrut
$ wo 'Collect[x*y + x*z, x]'
x*(y + z)
```


## ExpandAll

```scrut
$ wo 'ExpandAll[x*(x + 1)^2]'
x + 2*x^2 + x^3
```

- [`D`](symbolic/D.md)
- [`Integrate`](symbolic/Integrate.md)
- [`Sum`](symbolic/Sum.md)
- [`Product`](symbolic/Product.md)
- [`Simplify`](symbolic/Simplify.md)
- [`FullSimplify`](symbolic/FullSimplify.md)
- [`Solve`](symbolic/Solve.md)
- [`NSolve`](symbolic/NSolve.md)
- [`Reduce`](symbolic/Reduce.md)
- [`SeriesCoefficient`](symbolic/SeriesCoefficient.md)
- [`FindRoot`](symbolic/FindRoot.md)
- [`DSolve`](symbolic/DSolve.md)
- [`Grad`](symbolic/Grad.md)
- [`Div`](symbolic/Div.md)
- [`Curl`](symbolic/Curl.md)
- [`Laplacian`](symbolic/Laplacian.md)
- [`FourierTransform`](symbolic/FourierTransform.md)
- [`LaplaceTransform`](symbolic/LaplaceTransform.md)
- [`InverseLaplaceTransform`](symbolic/InverseLaplaceTransform.md)

## Additional Functions

- [`ArcCurvature`](symbolic/ArcCurvature.md)
- [`AsymptoticIntegrate`](symbolic/AsymptoticIntegrate.md)
- [`CoefficientRules`](symbolic/CoefficientRules.md)
- [`Cyclotomic`](symbolic/Cyclotomic.md)
- [`Decompose`](symbolic/Decompose.md)
- [`Derivative`](symbolic/Derivative.md)
- [`Distribute`](symbolic/Distribute.md)
- [`Dt`](symbolic/Dt.md)
- [`ExpandDenominator`](symbolic/ExpandDenominator.md)
- [`ExpandNumerator`](symbolic/ExpandNumerator.md)
- [`ExponentialGeneratingFunction`](symbolic/ExponentialGeneratingFunction.md)
- [`FactorList`](symbolic/FactorList.md)
- [`FactorSquareFree`](symbolic/FactorSquareFree.md)
- [`FactorSquareFreeList`](symbolic/FactorSquareFreeList.md)
- [`FactorTerms`](symbolic/FactorTerms.md)
- [`FactorTermsList`](symbolic/FactorTermsList.md)
- [`FindArgMax`](symbolic/FindArgMax.md)
- [`FindArgMin`](symbolic/FindArgMin.md)
- [`FindInstance`](symbolic/FindInstance.md)
- [`FunctionDomain`](symbolic/FunctionDomain.md)
- [`FunctionExpand`](symbolic/FunctionExpand.md)
- [`GeneratingFunction`](symbolic/GeneratingFunction.md)
- [`InterpolatingPolynomial`](symbolic/InterpolatingPolynomial.md)
- [`InverseFourierTransform`](symbolic/InverseFourierTransform.md)
- [`MaxLimit`](symbolic/MaxLimit.md)
- [`Maximize`](symbolic/Maximize.md)
- [`MinLimit`](symbolic/MinLimit.md)
- [`MinimalPolynomial`](symbolic/MinimalPolynomial.md)
- [`Minimize`](symbolic/Minimize.md)
- [`MonomialList`](symbolic/MonomialList.md)
- [`NMaxValue`](symbolic/NMaxValue.md)
- [`NMinValue`](symbolic/NMinValue.md)
- [`PolynomialMod`](symbolic/PolynomialMod.md)
- [`PolynomialQ`](symbolic/PolynomialQ.md)
- [`PolynomialQuotientRemainder`](symbolic/PolynomialQuotientRemainder.md)
- [`Refine`](symbolic/Refine.md)
- [`Root`](symbolic/Root.md)
- [`Roots`](symbolic/Roots.md)
- [`SolveAlways`](symbolic/SolveAlways.md)
- [`SymmetricPolynomial`](symbolic/SymmetricPolynomial.md)
- [`ToRadicals`](symbolic/ToRadicals.md)
- [`ToRules`](symbolic/ToRules.md)
- [`Tuples`](symbolic/Tuples.md)
- [`Wronskian`](symbolic/Wronskian.md)
- [`DSolveValue`](symbolic/DSolveValue.md)
- [`Interpolation`](symbolic/Interpolation.md)
- [`FindFit`](symbolic/FindFit.md)
- [`FindClusters`](symbolic/FindClusters.md)
- [`NIntegrate`](symbolic/NIntegrate.md)
- [`MatrixExp`](symbolic/MatrixExp.md)
- [`AsymptoticSolve`](symbolic/AsymptoticSolve.md)
- [`CoefficientArrays`](symbolic/CoefficientArrays.md)
- [`DifferenceDelta`](symbolic/DifferenceDelta.md)
- [`DifferenceQuotient`](symbolic/DifferenceQuotient.md)
- [`DiscreteConvolve`](symbolic/DiscreteConvolve.md)
- [`DiscreteLimit`](symbolic/DiscreteLimit.md)
- [`Eliminate`](symbolic/Eliminate.md)
- [`ExpandAll`](symbolic/ExpandAll.md)
- [`FindMaximum`](symbolic/FindMaximum.md)
- [`FindMinimum`](symbolic/FindMinimum.md)
- [`Fourier`](symbolic/Fourier.md)
- [`FourierCosTransform`](symbolic/FourierCosTransform.md)
- [`FourierSinTransform`](symbolic/FourierSinTransform.md)
- [`FrenetSerretSystem`](symbolic/FrenetSerretSystem.md)
- [`FromCoefficientRules`](symbolic/FromCoefficientRules.md)
- [`HornerForm`](symbolic/HornerForm.md)
- [`InverseFourier`](symbolic/InverseFourier.md)
- [`InverseFunction`](symbolic/InverseFunction.md)
- [`ListInterpolation`](symbolic/ListInterpolation.md)
- [`NDSolve`](symbolic/NDSolve.md)
- [`NDSolveValue`](symbolic/NDSolveValue.md)
- [`NMaximize`](symbolic/NMaximize.md)
- [`NMinimize`](symbolic/NMinimize.md)
- [`RSolve`](symbolic/RSolve.md)
- [`RecurrenceTable`](symbolic/RecurrenceTable.md)
- [`Series`](symbolic/Series.md)
