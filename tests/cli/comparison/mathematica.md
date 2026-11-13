---
icon: lucide/scale
---

# Comparison with Mathematica

[WolframScript] is the official command line interface for the Wolfram Language
provided by [Wolfram Research], the company behind the Wolfram Language.

[Mathematica] is the official frontend with a notebook interface.
It is implemented as a cross-platform desktop application
and is available for macOS, Linux, and Windows.

[WolframScript]: https://www.wolfram.com/wolframscript/
[Wolfram Research]: https://www.wolfram.com
[Mathematica]: https://www.wolfram.com/mathematica/

<dl>
  <dt>Implementation Language</dt>
  <dd>C++</dd>
  <dt>First Release</dt>
  <dd>1988</dd>
  <dt>License</dt>
  <dd>Proprietary</dd>
</dl>

Woxi is our alternative to WolframScript and
Woxi Studio is our alternative to Mathematica.
They try to be as compatible as possible, but there are a few features,
they intentionally deviate from to provide a better user experience.


## WolframScript vs Woxi

Woxi can do several things WolframScript and the Wolfram Engine can't,
like running without license checks, being embedded into other software,
running in the browser, and executing `.nb` notebooks directly.
See [What Woxi can do that Mathematica can't](mathematica/woxi_only.md)
for the full list.


## Mathematica vs Woxi Studio

- **Woxi Studio does not support out of order evaluation of cells** \
    When running a cell, it automatically also runs all cells before it.
    This is to avoid confusion about the state of the kernel
    and ensures consistent results when working with notebooks.

- **Woxi Studio does not support `%`** \
    This is too brittle as it refers to the last calculation that was evaluated,
    which could be any notebook cell and therefore leads to confusion
    about the state of the kernel.
    If you want to reuse results, assign them to a variable.

- **Mostly not implemented yet**
    - [Wolfram Knowledgebase](https://www.wolfram.com/language/core-areas/knowledgebase/) \
        This includes functions like:
        - `WolframAlpha[]`
        - Built-in `Entity[]` objects
        - Natural language input with `ctrl =`
        - Most functions listed on
            http://reference.wolfram.com/language/guide/KnowledgeRepresentationAndAccess.html
    - [Machine Learning and Neural Networks](https://www.wolfram.com/language/core-areas/machine-learning/)
    - [Optimization](https://www.wolfram.com/language/core-areas/optimization/)
    - [FEM](https://www.wolfram.com/language/core-areas/fem/)
    - [Chemistry](https://www.wolfram.com/language/core-areas/chemistry/)
    - [Audio Computation](https://www.wolfram.com/language/core-areas/audio/)
    - [Video Computation](https://www.wolfram.com/language/core-areas/video/)
    - [Astronomy](https://www.wolfram.com/language/core-areas/astronomy/)
    - [Control Systems](https://www.wolfram.com/language/core-areas/controls/)
    - [Signal Processing](https://www.wolfram.com/language/core-areas/signal/)
    - [Tools for AIs](https://www.wolfram.com/artificial-intelligence/)


## Missing features by Mathematica release

Woxi targets a *subset* of the Wolfram Language, so some of the
several-hundred functions added in each
[Mathematica release](https://writings.stephenwolfram.com/version-release/)
are not (yet) implemented.
[The full list](mathematica/missing_features.md) highlights the marquee
feature areas of each version that Woxi does **not** support.


## Conformance gaps

Where a function *is* implemented, its output should match `wolframscript`
character for character.
[The known exceptions](mathematica/conformance_gaps.md) are cataloged
separately — each one verified against the Wolfram Language, but not fixed
yet.
