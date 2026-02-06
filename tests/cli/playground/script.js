import {
  EditorView, keymap, lineNumbers, highlightSpecialChars, drawSelection,
  highlightActiveLine, rectangularSelection, crosshairCursor,
} from "https://esm.sh/@codemirror/view@6"
import { EditorState, Compartment } from "https://esm.sh/@codemirror/state@6"
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


let worker = null
let editorView = null

const wolframLanguage = StreamLanguage.define(mathematica)
const themeConfig = new Compartment()

function isDark() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
}

function getThemeExtension() {
  return isDark() ? oneDark : []
}

// Custom keybinding: Ctrl/Cmd+Enter to run
const runKeymap = keymap.of([{
  key: "Mod-Enter",
  run: () => {
    document.getElementById("runBtn").click()
    return true
  },
}])

// Initialize CodeMirror editor
editorView = new EditorView({
  doc: "Select[Range[30], PrimeQ]",
  extensions: [
    lineNumbers(),
    highlightSpecialChars(),
    history(),
    drawSelection(),
    EditorState.allowMultipleSelections.of(true),
    wolframLanguage,
    syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
    bracketMatching(),
    closeBrackets(),
    rectangularSelection(),
    crosshairCursor(),
    highlightActiveLine(),
    keymap.of([
      ...closeBracketsKeymap,
      ...defaultKeymap,
      ...historyKeymap,
    ]),
    runKeymap,
    EditorView.lineWrapping,
    themeConfig.of(getThemeExtension()),
  ],
  parent: document.getElementById("editor"),
})

// Switch CodeMirror theme when system preference changes
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  editorView.dispatch({
    effects: themeConfig.reconfigure(getThemeExtension()),
  })
})


function showStatus(message, type = "info") {
  const status = document.getElementById("status")

  if (message === "") {
    status.style.display = "none"
    return
  }

  status.textContent = message
  status.className = `status ${type}`
  status.style.display = "block"

  if (type === "info") {
    setTimeout(() => { status.style.display = "none" }, 3000)
  }
}

function showSpinner(id) {
  const el = document.getElementById(id)
  if (el) el.style.display = "block"
}

function hideSpinner(id) {
  const el = document.getElementById(id)
  if (el) el.style.display = "none"
}

function getEditorContent() {
  return editorView.state.doc.toString()
}

function setEditorContent(text) {
  editorView.dispatch({
    changes: {
      from: 0,
      to: editorView.state.doc.length,
      insert: text,
    },
  })
}

function initWorker() {
  showStatus("Loading Woxi WebAssembly module ...", "info")

  worker = new Worker("worker.js", { type: "module" })

  worker.onmessage = (e) => {
    const { type, success, message, result } = e.data

    if (type === "init") {
      if (success) {
        showStatus("")
        document.getElementById("runBtn").disabled = false
      }
      else {
        showStatus("Failed to load Woxi: " + message, "error")
      }
    }
    else if (type === "result") {
      hideSpinner("runSpinner")
      document.getElementById("runBtn").disabled = false

      if (success) {
        document.getElementById("output").textContent = result
      }
      else {
        document.getElementById("output").textContent = message
      }
    }
  }

  worker.onerror = (error) => {
    showStatus("Worker error: " + error.message, "error")
    hideSpinner("runSpinner")
    document.getElementById("runBtn").disabled = false
  }

  worker.postMessage({ type: "init" })
}

// Run button
document.getElementById("runBtn").addEventListener("click", () => {
  const code = getEditorContent().trim()
  if (!code || !worker) return

  document.getElementById("runBtn").disabled = true
  showSpinner("runSpinner")
  document.getElementById("output").textContent = ""

  worker.postMessage({ type: "evaluate", code: code })
})

// Clear button
document.getElementById("clearBtn").addEventListener("click", () => {
  setEditorContent("")
  document.getElementById("output").textContent = ""

  if (worker) {
    worker.postMessage({ type: "clear" })
  }
})

// Example buttons
document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    setEditorContent(btn.dataset.code)
  })
})

initWorker()
