# Comparison

## WolframScript & Mathematica

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


## Woxi

Woxi is our alternative to WolframScript and
Woxi Studio is our alternative to Mathematica.

<dl>
  <dt>Implementation Language</dt>
  <dd>Rust</dd>
  <dt>First Release</dt>
  <dd>2024</dd>
  <dt>License</dt>
  <dd>AGPL-3.0-or-later</dd>
</dl>

They try to be as compatible as possible, but there are a few features,
they intentionally deviate from the official Wolfram Language implementations
to provide a better user experience.


### WolframScript vs Woxi

- **Woxi supports Unicode characters** \
    For example to calculate the circumference of a circle with radius 4:
    ```sh
    woxi eval 'N[2π * 4]'
    ```


### Mathematica vs Woxi Studio

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
    - [Geography](https://www.wolfram.com/language/core-areas/geography/)
    - [Astronomy](https://www.wolfram.com/language/core-areas/astronomy/)
    - [Control Systems](https://www.wolfram.com/language/core-areas/controls/)
    - [Signal Processing](https://www.wolfram.com/language/core-areas/signal/)
    - [Tools for AIs](https://www.wolfram.com/artificial-intelligence/)


## Mathics

[Mathics] is the only major open source implementation of the Wolfram Language
apart from Woxi.

[Mathics]: https://mathics.org

<dl>
  <dt>Implementation Language</dt>
  <dd>Python</dd>
  <dt>First Release</dt>
  <dd>2011</dd>
  <dt>License</dt>
  <dd>GPL-3.0</dd>
</dl>
