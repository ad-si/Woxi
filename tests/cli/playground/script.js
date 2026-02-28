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
import LZString from "https://esm.sh/lz-string@1"


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
const STORAGE_KEY_OUTPUTS = "woxi-playground-outputs"
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
    runKeymap,
    keymap.of([
      ...closeBracketsKeymap,
      ...defaultKeymap,
      ...historyKeymap,
    ]),
    persistenceListener,
    EditorView.lineWrapping,
    themeConfig.of(getThemeExtension()),
  ],
  parent: document.getElementById("editor"),
})

// Restore code from shared URL
const params = new URLSearchParams(window.location.search)
const sharedCode = params.get("code")
if (sharedCode) {
  try {
    const decoded = LZString.decompressFromEncodedURIComponent(sharedCode)
    if (decoded) {
      setEditorContent(decoded)
    }
  } catch (_) { /* ignore corrupt share links */ }
  // Clean the URL without reloading
  window.history.replaceState(null, "", window.location.pathname)
}

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
  const outputsEl = document.getElementById("outputs")
  localStorage.setItem(STORAGE_KEY_OUTPUTS, outputsEl.innerHTML)
}

function restoreOutput() {
  try {
    const html = localStorage.getItem(STORAGE_KEY_OUTPUTS)
    if (html) {
      document.getElementById("outputs").innerHTML = html
    }
  } catch (_) { /* ignore corrupt data */ }
}

restoreOutput()

function renderOutputItems(items) {
  const outputsEl = document.getElementById("outputs")
  outputsEl.innerHTML = ""

  for (const item of items) {
    if (item.type === "graphics") {
      const div = document.createElement("div")
      div.className = "output-box graphics-box"
      div.innerHTML = item.svg
      outputsEl.appendChild(div)
    } else if (item.type === "text") {
      if (item.svg) {
        const div = document.createElement("div")
        div.className = "output-box text-box"
        div.innerHTML = item.svg
        outputsEl.appendChild(div)
      } else {
        const pre = document.createElement("pre")
        pre.className = "output-box text-box"
        pre.textContent = item.text
        outputsEl.appendChild(pre)
      }
    } else if (item.type === "print") {
      const pre = document.createElement("pre")
      pre.className = "output-box print-box"
      pre.textContent = item.text
      outputsEl.appendChild(pre)
    } else if (item.type === "warning") {
      const pre = document.createElement("pre")
      pre.className = "output-box warning-box"
      pre.textContent = item.text
      outputsEl.appendChild(pre)
    } else if (item.type === "error") {
      const pre = document.createElement("pre")
      pre.className = "output-box error-box"
      pre.textContent = item.text
      outputsEl.appendChild(pre)
    }
  }
}

function clearOutputs() {
  document.getElementById("outputs").innerHTML = ""
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
        try {
          const items = JSON.parse(result)
          renderOutputItems(items)
        } catch (_) {
          renderOutputItems([{ type: "error", text: result }])
        }
      }
      else {
        renderOutputItems([{ type: "error", text: message }])
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
  clearOutputs()

  worker.postMessage({ type: "evaluate", code: code })
})

// Clear button
document.getElementById("clearBtn").addEventListener("click", () => {
  setEditorContent("")
  localStorage.removeItem(STORAGE_KEY)
  localStorage.removeItem(STORAGE_KEY_OUTPUTS)
  clearOutputs()

  if (worker) {
    worker.postMessage({ type: "clear" })
  }
})

// Share button
document.getElementById("shareBtn").addEventListener("click", () => {
  const code = getEditorContent().trim()
  if (!code) return

  const compressed = LZString.compressToEncodedURIComponent(code)
  const url = new URL(window.location.pathname, window.location.origin)
  url.searchParams.set("code", compressed)

  navigator.clipboard.writeText(url.toString()).then(() => {
    showStatus("Link copied to clipboard", "info")
  }, () => {
    // Fallback: select the URL in a prompt
    prompt("Copy this link to share:", url.toString())
  })
})

// Example buttons
document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    setEditorContent(btn.dataset.code)
    localStorage.removeItem(STORAGE_KEY_OUTPUTS)
    clearOutputs()
  })
})

initWorker()
