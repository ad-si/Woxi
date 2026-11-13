---
icon: lucide/sparkles
---

# What Woxi Can Do That Mathematica Can't

Woxi implements only a subset of the Wolfram Language
(see [missing features](missing_features.md)),
but being open source, self-contained, and written in Rust
lets it do several things that Mathematica, WolframScript,
and the Wolfram Engine can't.


## Licensing & cost

- **No license checks** \
    Woxi never activates, phones home, or validates a license.
    It works offline, in CI pipelines, in containers,
    and on air-gapped machines without any setup.

- **Free of charge** \
    Mathematica is a paid product.
    Woxi, Woxi Studio, the [Playground](https://woxi.ad-si.com/playground/),
    and [JupyterLite](../../jupyter.md) are free,
    including for commercial use under the terms of the AGPL.

- **Unlimited parallel instances** \
    A Wolfram license caps the number of kernels that may run at once.
    Starting another `wolframscript` while the slots are taken fails with
    a misleading "not activated" error.
    You can start as many `woxi` processes as your machine can handle,
    e.g. to evaluate a batch of scripts in parallel:
    ```sh
    ls *.wls | xargs -P 8 -n 1 woxi run
    ```

- **Open source** \
    Every algorithm can be read, audited, debugged, and fixed.
    Results don't depend on a black box.


## Embedding

- **Embed it in other software** \
    Calling the Wolfram Language from another program
    (via WSTP, LibraryLink, or the Wolfram Client Libraries)
    requires a separately installed and licensed Wolfram Engine
    running as its own process,
    and shipping it with your product requires a commercial license.
    Woxi runs in-process as a library
    and can be bundled with your application:
    - Rust: the `woxi` crate (`woxi::interpret("Plus[1, 2]")`)
    - Python: `pip install woxi` (see [Python](../../python.md))
    - JavaScript / WebAssembly: the `woxi-wasm` npm package

- **Runs in the browser** \
    Compiled to WebAssembly, Woxi evaluates code entirely client-side
    in the [Playground](https://woxi.ad-si.com/playground/)
    and in [JupyterLite](../../jupyter.md) — no server, no account.

- **Shareable links** \
    The Playground encodes your code in the URL,
    so you can share a runnable snippet with a link.
    For example, this one evaluates `Range[9]`:
    <https://woxi.ad-si.com/playground/?code=EoQwdg5gpg2gnAXSA>

- **Custom builds** \
    Since the full source is available,
    you can compile your own version of Woxi containing only the
    functions you need, which results in a much smaller binary.

- **Instant startup** \
    `woxi eval` starts and finishes in a few milliseconds,
    while launching a Wolfram kernel takes most of a second or more.
    This makes Woxi practical for shell pipelines
    and for running many small scripts.


## Language & file formats

- **Unicode characters in input** \
    For example to calculate the circumference of a circle with radius 4:
    ```sh
    woxi eval 'N[2π * 4]'
    ```

- **Runs `.nb` notebooks directly** \
    WolframScript can only execute `.wls` / `.m` script files, but Woxi can
    evaluate the Input/Code cells of a Mathematica `.nb` notebook right from
    the command line:
    ```sh
    woxi run notebook.nb
    ```

- **Imports [CERN ROOT](https://root.cern/) files** \
    The Wolfram Language has no importer for the standard container format
    of particle physics, but Woxi decodes its 1-D and 2-D histograms,
    strings, and nested directories into an Association, and reads TTree
    branch data — flat leaves, leaf-count arrays, `std::vector`s, and
    `TLorentzVector`s — through element paths
    (see [`Import`](../../file_system/Import.md#cern-root-files-woxi-extension)):
    ```sh
    woxi eval 'Import["experiment.root"]'
    woxi eval 'Import["experiment.root", {"ROOT", "events", "energy"}]'
    ```
