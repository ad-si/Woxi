# Woxi

A Rust-based interpreter for a subset of the Wolfram Language.


## Features

The initial focus is to implement a subset of the Wolfram Language
so that it can be used for CLI scripting and Jupyter notebooks.
For example:

```wolfram
#!/usr/bin/env woxi

(* Print 5 random integers between 1 and 6 *)
Print[RandomInteger[{1, 6}, 5]]
```

All code examples in this documentation are run using Woxi.

Woxi runs faster than WolframScript as there is no overhead of starting a kernel
and verifying its license.
