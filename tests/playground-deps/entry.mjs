// Entry point bundled by esbuild into ../playground/vendor/codemirror.js.
// Re-exports exactly the CodeMirror (and LZString) symbols the playground's
// script.js imports, so the bundle contains only what is actually used.
//
// To regenerate the bundle after changing versions or the export list:
//   make playground-codemirror   (or: cd tests/playground-deps && npm ci && npm run build)

export {
  EditorView,
  keymap,
  lineNumbers,
  highlightSpecialChars,
  highlightActiveLine,
} from "@codemirror/view"

export { EditorState, Compartment } from "@codemirror/state"

export {
  StreamLanguage,
  syntaxHighlighting,
  defaultHighlightStyle,
  bracketMatching,
} from "@codemirror/language"

export {
  defaultKeymap,
  history,
  historyKeymap,
  indentWithTab,
} from "@codemirror/commands"

export { closeBrackets, closeBracketsKeymap } from "@codemirror/autocomplete"

export { mathematica } from "@codemirror/legacy-modes/mode/mathematica"

export { oneDark } from "@codemirror/theme-one-dark"

// lz-string is a CommonJS module with a default export; re-export it as a
// named binding so the playground can `import { LZString }`.
export { default as LZString } from "lz-string"
