import {
  EditorView, keymap, lineNumbers, highlightSpecialChars, drawSelection,
} from "https://esm.sh/@codemirror/view@6"
import { EditorState } from "https://esm.sh/@codemirror/state@6"
import {
  StreamLanguage,
  syntaxHighlighting, defaultHighlightStyle, bracketMatching,
} from "https://esm.sh/@codemirror/language@6"
import { defaultKeymap, history, historyKeymap } from "https://esm.sh/@codemirror/commands@6"
import {
  closeBrackets, closeBracketsKeymap,
} from "https://esm.sh/@codemirror/autocomplete@6"
import {
  mathematica,
} from "https://esm.sh/@codemirror/legacy-modes@6/mode/mathematica"
import { oneDark } from "https://esm.sh/@codemirror/theme-one-dark@6"

const wolframLanguage = StreamLanguage.define(mathematica)

function isDark() {
  return document.documentElement.classList.contains("dark")
}

const compactTheme = EditorView.theme({
  "&": { fontSize: "0.85em" },
  ".cm-content": { padding: "0.75rem 0", fontFamily: "ui-monospace, monospace" },
  ".cm-line": { padding: "0 0.75rem" },
  ".cm-scroller": { overflow: "auto" },
})

const lightTheme = EditorView.theme({
  "&": { backgroundColor: "#fafafa" },
  ".cm-gutters": { backgroundColor: "#fafafa", borderRight: "1px solid #e5e7eb" },
})

const readOnlyTheme = EditorView.theme({
  "&": { fontSize: "0.85em" },
  ".cm-content": { padding: "0.75rem 0", fontFamily: "ui-monospace, monospace" },
  ".cm-line": { padding: "0 0.75rem" },
  ".cm-scroller": { overflow: "auto" },
  ".cm-cursor": { display: "none" },
  "&.cm-focused": { outline: "none" },
})

const readOnlyLightTheme = EditorView.theme({
  "&": { backgroundColor: "#fafafa" },
})

function themeExtension() {
  return isDark() ? oneDark : lightTheme
}

function readOnlyThemeExtension() {
  return isDark() ? oneDark : readOnlyLightTheme
}

/**
 * Create a read-only CodeMirror view for displaying code.
 * No line numbers, no cursor, not focusable.
 */
export function createReadOnlyEditor(parent, code) {
  return new EditorView({
    doc: code.replace(/\n+$/, ""),
    extensions: [
      wolframLanguage,
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      EditorView.lineWrapping,
      EditorState.readOnly.of(true),
      EditorView.editable.of(false),
      readOnlyThemeExtension(),
      readOnlyTheme,
    ],
    parent,
  })
}

/**
 * Create an editable CodeMirror editor.
 * `onRun` fires on Ctrl/Cmd+Enter, `onCancel` on Escape.
 */
export function createEditor(parent, code, { onRun, onCancel } = {}) {
  const keybindings = []
  if (onRun) {
    keybindings.push({ key: "Mod-Enter", run: () => { onRun(); return true } })
  }
  if (onCancel) {
    keybindings.push({ key: "Escape", run: () => { onCancel(); return true } })
  }

  return new EditorView({
    doc: code,
    extensions: [
      lineNumbers(),
      highlightSpecialChars(),
      history(),
      drawSelection(),
      wolframLanguage,
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      bracketMatching(),
      closeBrackets(),
      EditorView.lineWrapping,
      themeExtension(),
      compactTheme,
      keymap.of([
        ...keybindings,
        ...closeBracketsKeymap,
        ...defaultKeymap,
        ...historyKeymap,
      ]),
    ],
    parent,
  })
}

export function getContent(view) {
  return view.state.doc.toString()
}

export function destroyEditor(view) {
  view.destroy()
}
