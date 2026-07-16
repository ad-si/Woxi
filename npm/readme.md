# Woxi

An interpreter for a subset of the
[Wolfram Language](https://www.wolfram.com/language/)
powered by Rust and compiled to WebAssembly.

This package wraps the [Woxi](https://github.com/ad-si/Woxi) interpreter
so it can be used from Node.js — no Wolfram installation required.
Woxi is a computer algebra system (CAS): computations are solved symbolically.


## Installation

```sh
npm install woxi-wasm
```

(The package is named `woxi-wasm` because the name `woxi` is not
available on npm.)


## Usage

```js
import { evaluate } from "woxi-wasm"

evaluate("Plus[1, 2]")        //=> "3"
evaluate("1/3 + 1/6")         //=> "1/2"
evaluate("Sqrt[8]")           //=> "2*Sqrt[2]"
evaluate('StringReverse["hello"]')  //=> "olleh"
```

CommonJS works too:

```js
const { evaluate } = require("woxi-wasm")
```

Interpreter state (variables, function definitions) persists across calls:

```js
import woxi from "woxi-wasm"

woxi.evaluate("f[x_] := x^2")
woxi.evaluate("f[5]")  //=> "25"
woxi.clear()           // reset all state
```


### Structured output

`evaluateAll` returns one item per output, including graphics as SVG:

```js
const items = woxi.evaluateAll('Print["hi"]\nGraphics[{Disk[]}]')
//=> [
//     { type: "print", text: "hi" },
//     { type: "graphics", svg: "<svg …" },
//   ]
```

Item types: `text`, `print`, `graphics`, `warning`, `error`, `sound`,
and `manipulate`.


### Virtual files

There is no filesystem access from WebAssembly, so `Import[…]` reads from
an in-memory store you populate first:

```js
woxi.setVirtualFile("data.csv", "a,b\n1,2\n")
woxi.evaluate('Import["data.csv"]')  //=> "{{a, b}, {1, 2}}"
```

`Import["https://…"]` fetches over HTTP automatically.


## API

| Function                     | Description                                       |
|------------------------------|---------------------------------------------------|
| `evaluate(code)`             | Evaluate code, return output as a string          |
| `evaluateAll(code)`          | Evaluate code, return structured output items     |
| `splitStatements(code)`      | Split code into top-level statements              |
| `evaluateStatement(stmt)`    | Evaluate a single statement (structured output)   |
| `getGraphics()`              | SVG captured by the last `evaluate()` call        |
| `getSound()`                 | Base64 audio captured by the last call            |
| `getWarnings()`              | Warnings of the last call as an array             |
| `clear()`                    | Clear all interpreter state                       |
| `setDarkMode(enabled)`       | Toggle dark-mode colors for SVG output            |
| `setVirtualFile(name, data)` | Register an in-memory file for `Import[…]`        |
| `clearVirtualFiles()`        | Remove all registered in-memory files             |

TypeScript definitions are included.


## Development

The wasm bundle in `pkg/` is generated from the Rust sources — build it from
the [repository](https://github.com/ad-si/Woxi) root:

```sh
make npm-build   # build pkg/ with wasm-pack
make npm-test    # build + run the JS test suite
```


## License

[AGPL-3.0-or-later](https://github.com/ad-si/Woxi/blob/main/license.txt)
