# TableForm — Examples

All examples extracted from the official Wolfram Language reference page:
<https://reference.wolfram.com/language/ref/TableForm.html>

> **Note:** On the original page the outputs (`Out[n]=`) are rendered as typeset
> images/graphics, so only the input code is reproducible as text. Output labels
> are noted where the page shows them. Inputs marked *(pasted typeset output)*
> are the raw box-form expressions the page uses to demonstrate copying a
> typeset output back into an input cell.

## Basic Examples (1)

Show vector, matrix and general arrays in tabular form:

`In[1]:=`

```wolfram
TableForm[Array[a, {2}]]
```

`In[2]:=`

```wolfram
TableForm[Array[a, {2}]];
%
```

`Out[2]=`

`In[3]:=` *(pasted typeset output)*

```wolfram
\!\(
TagBox[
TagBox[GridBox[{
{
RowBox[{"a", "[", "1", "]"}]},
{
RowBox[{"a", "[", "2", "]"}]}
},
GridBoxAlignment->{"Columns" -> {{Left}}, "Rows" -> {{Baseline}}},
GridBoxSpacings->{"Columns" -> {
Offset[0.27999999999999997`], {
Offset[0.5599999999999999]}, 
Offset[0.27999999999999997`]}, "Rows" -> {
Offset[0.2], {
Offset[0.4]}, 
Offset[0.2]}}],
Column],
Function[BoxForm`e$, 
TableForm[BoxForm`e$]]]\)
```

`Out[3]=`

`In[4]:=`

```wolfram
TableForm[Array[a, {2, 2}]]
```

`In[5]:=`

```wolfram
TableForm[Array[a, {2, 2, 2}]]
```

## Scope (3)

Tables of numbers and formulas:

`In[1]:=`

```wolfram
TableForm[RandomReal[5, {3, 4}]]
```

`In[2]:=`

```wolfram
TableForm[Table[1/(i + j), {i, 4}, {j, 4}]]
```

`In[3]:=`

```wolfram
TableForm[RotationMatrix[\[Theta], {UnitVector[5, 2], UnitVector[5, 4]}]]
```

A table of lists:

`In[1]:=`

```wolfram
Outer[List, Range[4], Range[4]]
```

`Out[1]=`

`In[2]:=`

```wolfram
Outer[List, Range[4], Range[4]];
TableForm[%, TableDepth -> 2]
```

Format a ragged array:

`In[1]:=`

```wolfram
TableForm[Table[Range[i], {i, 5}]]
```

## Options (11)

### TableAlignments (3)

Specify the alignment of columns:

`In[1]:=`

```wolfram
TableForm[Partition[Range[15]!, 3], TableAlignments -> Center]
```

Define both horizontal and vertical alignments:

`In[1]:=`

```wolfram
TableForm[
 Table[Graphics[Rectangle[], 
   ImageSize -> (i + j)], {i, {10, 25, 40}}, {j, {10, 25, 40}}], 
 TableAlignments -> {Right, Center}]
```

Align columns on a decimal point or any character:

`In[1]:=`

```wolfram
TableForm[0.12345 * 10^Range[4], TableAlignments -> "."]
```

### TableDepth (2)

By default all dimensions are formatted:

`In[1]:=`

```wolfram
TableForm[{{a, b}, {c, d}, {e, f}}]
```

Only use tabular formatting for the outermost dimension:

`In[1]:=`

```wolfram
TableForm[{{a, b}, {c, d}, {e, f}}, TableDepth -> 1]
```

### TableDirections (2)

By default the outermost dimension is a column:

`In[1]:=`

```wolfram
TableForm[{{a, b}, {c, d}, {e, f}}]
```

Format the first dimension as a row instead:

`In[1]:=`

```wolfram
TableForm[{{a, b}, {c, d}, {e, f}}, TableDirections -> Row]
```

### TableHeadings (3)

Specify headings for rows:

`In[1]:=`

```wolfram
TableForm[{{a, b}, {c, d}, {e, f}}, 
 TableHeadings -> {{"r1", "r2", "r3"}, None}]
```

Specify headings for columns:

`In[1]:=`

```wolfram
TableForm[{{a, b}, {c, d}, {e, f}}, 
 TableHeadings -> {None, {"c1", "c2"}}]
```

Specify headings for rows and columns:

`In[1]:=`

```wolfram
TableForm[{{a, b}, {c, d}, {e, f}}, 
 TableHeadings -> {{"r1", "r2", "r3"}, {"c1", "c2"}}]
```

### TableSpacing (1)

The default automatic spacing:

`In[1]:=`

```wolfram
TableForm[{{a, b}, {c, d}, {e, f}}]
```

Explicitly specify the spacing between rows and between columns:

`In[2]:=`

```wolfram
TableForm[{{a, b}, {c, d}, {e, f}}, TableSpacing -> {5, 2}]
```

## Applications (4)

Display data in a formatted table:

`In[1]:=`

```wolfram
TableForm[{{5, 7}, {4, 2}, {10, 3}}, 
 TableHeadings -> {{"Group A", "Group B", "Group C"}, {"y1", "y2"}}]
```

Create a multiplication table:

`In[1]:=`

```wolfram
TableForm[Outer[Times, Range[10], Range[10]], 
 TableHeadings -> Automatic]
```

Multiplication table for the cyclic group *C*₅:

`In[1]:=`

```wolfram
c5 = {1, "A", "B", "C", "D"};
```

`In[2]:=`

```wolfram
TableForm[Table[RotateLeft[c5, n], {n, 0, 4}], 
 TableHeadings -> {c5, c5}, TableSpacing -> {1, 2}]
```

Create a table of graphics:

`In[1]:=`

```wolfram
TableForm@
 Table[SphericalPlot3D[
   Evaluate@Abs@SphericalHarmonicY[l, m, \[Theta], \[Phi]], {\[Theta],
     0, Pi}, {\[Phi], 0, 2 Pi}, PlotRange -> 0.6, Mesh -> None, 
   Boxed -> False, Axes -> None, ImageSize -> 100], {l, 0, 3}, {m, 0, 
   l}]
```

## Properties & Relations (7)

TableForm formats arrays in a tabular form:

`In[1]:=`

```wolfram
TableForm[{{a, b}, {c, d}, {e, f}}]
```

MatrixForm formats arrays using standard matrix formatting:

`In[2]:=`

```wolfram
MatrixForm[{{a, b}, {c, d}, {e, f}}]
```

Grid formats two-dimensional arrays as a grid:

`In[3]:=`

```wolfram
Grid[{{a, b}, {c, d}, {e, f}}]
```

`Out[3]=`

Use MatrixPlot to visualize the structure of large arrays:

`In[1]:=`

```wolfram
b[i_] := Table[Binomial[n, k], {n, 0, i}, {k, 0, i}]
```

`In[2]:=`

```wolfram
b[5] // TableForm
```

`In[3]:=`

```wolfram
MatrixPlot[b[99]]
```

`Out[3]=`

Use ArrayPlot to visualize the structure of large discrete arrays:

`In[1]:=`

```wolfram
b[i_] := CellularAutomaton[30, {{1}, 0}, i]
```

`In[2]:=`

```wolfram
TableForm[b[5]]
```

`In[3]:=`

```wolfram
ArrayPlot[b[100]]
```

`Out[3]=`

Use Style to affect the display of TableForm:

`In[1]:=`

```wolfram
b[i_] := Table[Binomial[n, k], {n, 0, i}, {k, 0, i}]
```

`In[2]:=`

```wolfram
Style[TableForm[b[20]], Tiny]
```

`Out[2]=`

`In[3]:=`

```wolfram
Style[TableForm[b[3]], {Large, Bold, Orange}]
```

`Out[3]=`

Use any number form such as ScientificForm or BaseForm to affect the display of numbers:

`In[1]:=`

```wolfram
ScientificForm[TableForm[RandomReal[10^5, {4, 4}]], 3]
```

`In[2]:=`

```wolfram
PaddedForm[BaseForm[TableForm[RandomInteger[{0, 127}, {4, 4}]], 2], 8,
  NumberPadding -> {"0", ""}]
```

The typeset form of TableForm[expr] is interpreted the same as expr when used in input:

`In[1]:=`

```wolfram
f[TableForm[{{1, 2}, {3, 4}}]]
```

`Out[1]=`

Copy the output and paste it into an input cell. The pasted typeset table (1 2 / 3 4) is interpreted as `{{1,2},{3,4}}`:

`In[2]:=` *(pasted typeset output)*

```wolfram
f[ \!\(\*
TagBox[GridBox[{
{"1", "2"},
{"3", "4"}
},
GridBoxAlignment->{"Columns" -> {{Left}}, "Rows" -> {{Baseline}}},
GridBoxSpacings->{"Columns" -> {
Offset[0.27999999999999997`], {
Offset[2.0999999999999996`]}, 
Offset[0.27999999999999997`]}, "Rows" -> {
Offset[0.2], {
Offset[0.4]}, 
Offset[0.2]}}],
Function[BoxForm`e$, 
TableForm[BoxForm`e$]]]\) ]
```

`Out[2]=`

When an input evaluates to TableForm[expr], TableForm does not appear in the output:

`In[1]:=`

```wolfram
TableForm[{{1, 2}, {3, 4}}]
```

Out is assigned the value {{1,2},{3,4}}, not TableForm[{{1,2},{3,4}}]:

`In[2]:=`

```wolfram
TableForm[{{1, 2}, {3, 4}}];
%
```

`Out[2]=`

## Possible Issues (1)

Even when an output omits TableForm from the top level, it is not stripped from subexpressions:

`In[1]:=`

```wolfram
e = TableForm[{{1, 2}, {3, 4}}]
```

The output does not have TableForm in it:

`In[2]:=`

```wolfram
e = TableForm[{{1, 2}, {3, 4}}];
%
```

`Out[2]=`

However, the variable e does have TableForm in it, which may affect subsequent evaluations:

`In[3]:=`

```wolfram
FullForm[e]
```

The determinant is not evaluated due to the intervening TableForm:

`In[4]:=`

```wolfram
Det[e]
```

`Out[4]=`

Assign variables first and then apply TableForm to the result to maintain computability:

`In[5]:=`

```wolfram
(f = {{1, 2}, {3, 4}}) // TableForm
```

`In[6]:=`

```wolfram
Det[f]
```

`Out[6]=`

## Neat Examples (1)

A Stirling table texture:

`In[1]:=`

```wolfram
Style[TableForm[
  Table[StirlingS2[n, k], {n, 1, 28}, {k, 1, 28}]], 2, Orange]
```

`Out[1]=`
