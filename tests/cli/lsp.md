---
icon: lucide/file-code-2
---

# Language Server

Woxi implements the [Language Server Protocol] (LSP) for the Wolfram
Language, so any editor with an LSP client can check and explore Woxi
scripts:

```sh
woxi lsp
```

The server communicates over stdin/stdout, which is how editors start it.
`--stdio` is accepted and ignored, since many clients pass it by
convention.

[Language Server Protocol]: https://microsoft.github.io/language-server-protocol/


## Features

Diagnostics
: Syntax errors, and warnings naming every symbol this interpreter does
  not implement — so you see which parts of a script Woxi will not run
  before running it. Symbols the file defines itself shadow the built-in
  and are never warned about.

Hover
: The description of the symbol under the cursor, its implementation
  status and a link to its documentation. For a symbol defined in the
  file, its definitions are shown instead.

Completion
: Every built-in symbol, with the ones Woxi implements ranked first, plus
  the symbols defined in the edited file.

Go to definition, find references, document highlight
: Resolved from the assignments in the file.

Document outline
: The file's top-level definitions. Locals bound inside `Module`, `Block`
  or `With` are left out of the outline, but are still found by
  go to definition.

Hover, completion and navigation are computed from a forgiving tokenizer
rather than the grammar, so they keep working while a file is mid-edit
and does not parse yet.


## Editor Configuration

Point your editor at `woxi lsp` for `.wl`, `.wls`, `.m` and `.nb` files.

[Neovim]'s built-in client:

```lua
vim.lsp.config.woxi = {
  cmd = { "woxi", "lsp" },
  filetypes = { "wolfram" },
  root_markers = { ".git" },
}
vim.lsp.enable("woxi")
```

[Helix] (`languages.toml`):

```toml
[language-server.woxi]
command = "woxi"
args = ["lsp"]

[[language]]
name = "wolfram"
scope = "source.wolfram"
file-types = ["wl", "wls", "m", "nb"]
roots = [".git"]
language-servers = ["woxi"]
```

[Neovim]: https://neovim.io
[Helix]: https://helix-editor.com
