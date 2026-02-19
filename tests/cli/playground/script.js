import {
  EditorView, keymap, lineNumbers, highlightSpecialChars,
  highlightActiveLine,
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

// Custom keybinding: Shift+Enter to run
const runKeymap = keymap.of([{
  key: "Shift-Enter",
  run: () => {
    document.getElementById("runBtn").click()
    return true
  },
}])

const STORAGE_KEY = "woxi-playground-code"
const STORAGE_KEY_OUTPUT = "woxi-playground-output"
const STORAGE_KEY_GRAPHICS = "woxi-playground-graphics"
const DEFAULT_CODE = "Select[Range[30], PrimeQ]"

function saveEditorContent() {
  const content = editorView.state.doc.toString()
  localStorage.setItem(STORAGE_KEY, content)
}

const persistenceListener = EditorView.updateListener.of((update) => {
  if (update.docChanged) saveEditorContent()
})

// Initialize CodeMirror editor
editorView = new EditorView({
  doc: localStorage.getItem(STORAGE_KEY) ?? DEFAULT_CODE,
  extensions: [
    lineNumbers(),
    highlightSpecialChars(),
    history(),
    wolframLanguage,
    syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
    bracketMatching(),
    closeBrackets(),
    highlightActiveLine(),
    keymap.of([
      ...closeBracketsKeymap,
      ...defaultKeymap,
      ...historyKeymap,
    ]),
    runKeymap,
    persistenceListener,
    EditorView.lineWrapping,
    themeConfig.of(getThemeExtension()),
  ],
  parent: document.getElementById("editor"),
})

editorView.focus()

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

function saveOutput() {
  const outputEl = document.getElementById("output")
  const graphicsEl = document.getElementById("graphics")
  localStorage.setItem(STORAGE_KEY_OUTPUT, JSON.stringify({
    html: outputEl.innerHTML,
    display: outputEl.style.display,
  }))
  localStorage.setItem(STORAGE_KEY_GRAPHICS, JSON.stringify({
    html: graphicsEl.innerHTML,
    display: graphicsEl.style.display,
  }))
}

function restoreOutput() {
  try {
    const output = JSON.parse(localStorage.getItem(STORAGE_KEY_OUTPUT))
    const graphics = JSON.parse(localStorage.getItem(STORAGE_KEY_GRAPHICS))
    if (output) {
      const el = document.getElementById("output")
      el.innerHTML = output.html
      el.style.display = output.display
    }
    if (graphics) {
      const el = document.getElementById("graphics")
      el.innerHTML = graphics.html
      el.style.display = graphics.display
    }
  } catch (_) { /* ignore corrupt data */ }
}

restoreOutput()

function initWorker() {
  showStatus("Loading Woxi WebAssembly module ...", "info")

  worker = new Worker("worker.js", { type: "module" })

  worker.onmessage = (e) => {
    const { type, success, message, result, warnings, graphics, outputSvg } = e.data

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
      const graphicsEl = document.getElementById("graphics")
      const outputEl = document.getElementById("output")

      if (success) {
        let output = ""
        if (warnings) output += warnings + "\n"

        if (graphics) {
          // Graphics/Plot/Grid: show the captured SVG
          graphicsEl.innerHTML = graphics
          graphicsEl.style.display = "block"
          // Strip "-Graphics-" from text output when SVG is shown
          output += result
          output = output.replace(/-Graphics-/g, "").trim()
          if (output) {
            outputEl.textContent = output
            outputEl.style.display = "block"
          } else {
            outputEl.textContent = ""
            outputEl.style.display = "none"
          }
        } else if (outputSvg) {
          // Non-graphics: render the result as SVG (with superscripts etc.)
          graphicsEl.innerHTML = ""
          graphicsEl.style.display = "none"
          // Show Print output (stdout) as plain text, result as SVG
          const stdout = output.trim()
          if (stdout) {
            outputEl.textContent = stdout
          } else {
            outputEl.textContent = ""
          }
          outputEl.innerHTML = (stdout ? outputEl.innerHTML + "\n" : "")
            + outputSvg
          outputEl.style.display = "block"
        } else {
          graphicsEl.innerHTML = ""
          graphicsEl.style.display = "none"
          output += result
          outputEl.textContent = output
          outputEl.style.display = "block"
        }
      }
      else {
        outputEl.textContent = message
        graphicsEl.innerHTML = ""
        graphicsEl.style.display = "none"
      }
      saveOutput()
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
  document.getElementById("graphics").innerHTML = ""
  document.getElementById("graphics").style.display = "none"

  worker.postMessage({ type: "evaluate", code: code })
})

// Clear button
document.getElementById("clearBtn").addEventListener("click", () => {
  setEditorContent("")
  localStorage.removeItem(STORAGE_KEY)
  localStorage.removeItem(STORAGE_KEY_OUTPUT)
  localStorage.removeItem(STORAGE_KEY_GRAPHICS)
  document.getElementById("output").textContent = ""
  document.getElementById("graphics").innerHTML = ""
  document.getElementById("graphics").style.display = "none"

  if (worker) {
    worker.postMessage({ type: "clear" })
  }
})

// Example buttons
document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    setEditorContent(btn.dataset.code)
    localStorage.removeItem(STORAGE_KEY_OUTPUT)
    localStorage.removeItem(STORAGE_KEY_GRAPHICS)
    document.getElementById("output").textContent = ""
    document.getElementById("output").style.display = "none"
    document.getElementById("graphics").innerHTML = ""
    document.getElementById("graphics").style.display = "none"
  })
})

initWorker()
