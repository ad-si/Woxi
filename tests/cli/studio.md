# Woxi Studio

**Woxi Studio** is a native notebook editor for Wolfram Language
`.nb` files, built with the [Iced] GUI framework.
It offers a Mathematica-like cell-based editing experience,
runs expressions through the Woxi interpreter,
and renders graphics inline as SVG.

[Iced]: https://iced.rs


## Running

From a checkout of the repository:

```sh
cargo run -p woxi-studio
```

The last opened or saved notebook is remembered in
`~/.config/woxi-studio/last_file` and re-opened automatically on the
next launch, so the editor comes up where you left off.
Opening a notebook via `File → Open` spawns a new Studio window
instead of replacing the current one, so multiple notebooks can be
edited side by side.


## Cell types

Each cell has its own text editor and a cell-type selector (with a
[Lucide] icon) in the gutter.  The following styles are supported:

- **Heading cells**: `Title`, `Subtitle`, `Section`, `Subsection`,
  `Subsubsection`, `Chapter`, `Subchapter`
- **Text cells**: `Text`, `Item`, `Subitem`
- **Code cells**: `Input`, `Code`
- **Output cells**: `Output`, `Print` output

`Title` and `Subtitle` cells are rendered in bold with a colored
accent, and the dropdown menu flips upward when it would otherwise
be clipped by the bottom of the window.
Changing a cell's style clears any stale output it had.

`Chapter` and `Subchapter` cells get a collapse chevron in a
reserved slot left of the type picker.  Clicking it hides every
cell that follows up to the next same-or-higher-level heading.
The collapsed state is serialized as
`Cell["...", "Chapter", CellOpen -> False]` so it is restored when
the notebook is reopened.

[Lucide]: https://lucide.dev


## Evaluation

- **`Shift+Enter`** evaluates the focused cell.  If there is another
  cell below, the cursor moves to it; otherwise a new input cell is
  appended and scrolled into view.
- The green ▶ play button on the right side of a cell evaluates just
  that cell.
- The circle-play button at the left of the toolbar (**Eval All**)
  evaluates every input/code cell from top to bottom.
- Evaluating a single cell re-runs all preceding input cells first,
  so variable assignments and function definitions from earlier
  cells are always in scope.
- Each statement inside a cell is evaluated separately (matching the
  Playground and Jupyter behavior), so every result is shown — not
  just the final one.
- When an input cell is modified after evaluation, its output is
  grayed out to signal that it is stale.
- Warnings and error messages surface inline in the cell instead of
  being printed to the terminal.
- The `Null` symbol, trailing semicolons, and void operations
  (function definitions, `Clear`, …) produce no visible output.


## Graphics & interactive output

`Graphics`, `Graphics3D`, and `Plot[…]` expressions are rendered
inline as SVG and pre-rasterized to a bitmap at the display scale
factor, so scrolling stays smooth even for complex plots.
Double-clicking a graphic opens a fullscreen modal for closer
inspection; `Escape`, clicking the backdrop, or the Close button
dismiss it without losing the scroll position.

`Manipulate[expr, {u, umin, umax}, …]` is rendered as an interactive
widget with sliders and pick lists.  Option values such as
`Initialization :> …` are preserved and prepended to every
re-evaluation.

Output text is rendered in a read-only text editor, so it can be
selected and copied with the mouse.


## Table of contents

A TOC sidebar on the left lists every heading cell from `Title`
through `Subsubsection`, indented by level.  Clicking an entry
scrolls the notebook to the corresponding cell using the widget's
actual bounds for precise positioning.  The sidebar is auto-shown
for notebooks that contain any headings, and its width adapts to
both the longest entry and the current window width.


## Editing

The input/code cells use a Wolfram Language syntax highlighter that
colors comments, strings, numbers, function names, keywords,
operators, and patterns.

**Text manipulation**

- **`Tab` / `Shift+Tab`** — indent / unindent
- **`Cmd+/`** (or `Ctrl+/`) — toggle comments.  With no selection
  it comments the current line; within a single line it wraps only
  the selected text; across multiple lines it comments every line
  that is touched, using `(* … *)` syntax.
- Selecting text and typing `{`, `[`, `(`, `"`, or `'` wraps the
  selection in the matching pair instead of replacing it.
- **`Cmd+Z` / `Cmd+Shift+Z`** — per-cell undo / redo.
- **`Cmd+C` / `Cmd+V` / `Cmd+X` / `Cmd+A`** — clipboard and
  select-all work as expected.
- **`Ctrl+A` / `Ctrl+E`** — Emacs-style jump to line start / end
  (works on Linux, macOS, and Windows).
- **`Ctrl+D`** — forward-delete the character under the cursor.
- **`Ctrl+W`** — delete the previous word.

**Navigation**

- **`Up` / `Down`** arrow keys move the cursor between lines within
  a cell and jump to the neighboring cell (or the `+` divider
  between cells) when already at the first/last line.
- Pressing `Enter` on a focused divider inserts a new cell there.
- Dividers are visually highlighted when focused.


## Reordering cells

Hovering over the gutter (left of a cell, below the type selector)
reveals a grip handle.  Press and drag it to move the cell up or
down; a blue indicator line shows the drop target.


## Export

Notebooks can be exported from the toolbar's **Export as** dropdown
to the following formats:

- **Mathematica Notebook** (`.nb`)
- **Jupyter Notebook** (`.ipynb`)
- **Markdown** (`.md`)
- **LaTeX** (`.tex`)
- **Typst** (`.typ`)
- **PDF** (`.pdf`) — rendered via the SVG→PDF pipeline powered by
  [`svg2pdf`][svg2pdf], with headings, text, code blocks, output
  text, and graphics embedded as vector content.

The save dialog pre-fills the current notebook path with the
appropriate extension and does not change the file currently being
edited.

[svg2pdf]: https://crates.io/crates/svg2pdf


## File management

- **`Cmd+N` / `Ctrl+N`** — new notebook (in a new window)
- **`Cmd+O` / `Ctrl+O`** — open `.nb` file
- **`Cmd+S` / `Ctrl+S`** — save
- Closing a notebook with unsaved changes prompts with a
  **Save / Don't Save / Cancel** dialog.
- The window title shows the current file name, e.g.
  `Woxi Studio | showcase.nb`.


## Appearance

- **Theme**: `Auto` (default, follows the OS light/dark preference),
  `Light`, or `Dark`.  Fonts are embedded Atkinson Hyperlegible Next
  (sans) and Atkinson Hyperlegible Mono so output looks identical
  across platforms.
- **Preview mode** (eye icon, top right) hides the gutter,
  dividers, play buttons, and editor borders for distraction-free
  reading.
- Notebook content is limited to 800 px and centered; input and
  output are grouped inside a single rounded container separated by
  a thin rule.


> [!TIP]
> Woxi Studio is the native counterpart to the in-browser
> [Playground](playground/index.html) and
> [JupyterLite](jupyterlite-showcase.html) environments — the same
> interpreter powers all three, so expressions that work in one
> work in the others.
