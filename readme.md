<img src="./images/logo.png" alt="Wordmark of Woxi">

# Woxi

A Rust-based interpreter for a subset of the Wolfram Language.


## Features

- Parse and evaluate basic arithmetic expressions
- Support for addition and subtraction operations
- Handling of integer and floating-point numbers
- Error handling for invalid inputs
- CLI for easy interaction

Check out the [functions.csv](./functions.csv) file
for a list of all Wolfram Language functions and their implementation status.


## Installation

To use this Wolfram Language interpreter, you need to have Rust installed on your system.
If you don't have Rust installed yet, you can get it from
[rust-lang.org](https://www.rust-lang.org/tools/install).

Clone the repository and build the project:

```bash
git clone https://github.com/ad-si/Woxi
cd Woxi
cargo build --release
```


## Usage

You can use the interpreter directly from the command line:

```bash
cargo run -- "1 + 2"
```

This will output: `3`


## CLI Comparison With Wolframscript

Woxi | Wolframscript
--- | ---
`woxi eval "1 + 2"` | `wolframscript -code "1 + 2"`
`woxi run script.wls` | `wolframscript script.wls`
`woxi repl` | `wolframscript`


## Contributing

Contributions are welcome!
Please feel free to submit a Pull Request.


### Testing

To run the test suite:

```sh
cargo test
```


## Related

- [CodeParser] - Parse Wolfram Language as AST or CST.
- [MMA Clone] - Simple Wolfram Language clone in Haskell.
- [TS Wolfram] - Toy Wolfram language interpreter in TypeScript.
- [Wolfram Parser] - Wolfram parser in Rust.

[CodeParser]: https://github.com/WolframResearch/codeparser
[MMA Clone]: https://github.com/mrtwistor/mmaclone
[TS Wolfram]: https://github.com/coffeemug/ts-wolfram
[Wolfram Parser]: https://github.com/oovm/wolfram-parser
