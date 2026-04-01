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
const STORAGE_KEY_THEME = "woxi-playground-theme"
const THEME_MODES = ["auto", "light", "dark"]
const THEME_ICONS = { auto: "\u25D0", light: "\u2600", dark: "\u263E" }
const THEME_LABELS = { auto: "Auto", light: "Light", dark: "Dark" }

function systemDark() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
}

function isDark() {
  return document.documentElement.classList.contains("dark")
}

function getThemeExtension() {
  return isDark() ? oneDark : []
}

let currentMode = localStorage.getItem(STORAGE_KEY_THEME) ?? "auto"
if (!THEME_MODES.includes(currentMode)) currentMode = "auto"

function applyMode(mode) {
  currentMode = mode
  const dark = mode === "dark" || (mode === "auto" && systemDark())
  document.documentElement.classList.toggle("dark", dark)
  const btn = document.getElementById("themeBtn")
  btn.textContent = THEME_ICONS[mode]
  btn.dataset.tooltip = `Theme: ${THEME_LABELS[mode]}`
}

applyMode(currentMode)

document.getElementById("themeBtn").addEventListener("click", () => {
  // From auto, jump to the opposite of what's currently shown
  const next = currentMode === "auto"
    ? (isDark() ? "light" : "dark")
    : THEME_MODES[(THEME_MODES.indexOf(currentMode) + 1) % THEME_MODES.length]
  applyMode(next)
  if (next === "auto") {
    localStorage.removeItem(STORAGE_KEY_THEME)
  } else {
    localStorage.setItem(STORAGE_KEY_THEME, next)
  }
  editorView.dispatch({
    effects: themeConfig.reconfigure(getThemeExtension()),
  })
  if (worker) {
    worker.postMessage({ type: "set_theme", dark: isDark() })
  }
})

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
      clearOutputs()
      localStorage.removeItem(STORAGE_KEY_OUTPUTS)
    }
  } catch (_) { /* ignore corrupt share links */ }
  // Clean the URL without reloading
  window.history.replaceState(null, "", window.location.pathname)
}

editorView.focus()

// Follow system preference changes when in auto mode
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (currentMode !== "auto") return
  applyMode("auto")
  editorView.dispatch({
    effects: themeConfig.reconfigure(getThemeExtension()),
  })
  if (worker) {
    worker.postMessage({ type: "set_theme", dark: isDark() })
  }
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
        // Resize SVG to match actual rendered text dimensions
        const svg = div.querySelector("svg")
        const text = svg && svg.querySelector("text")
        if (text) {
          const bbox = text.getBBox()
          svg.setAttribute("width", Math.ceil(bbox.x + bbox.width + 2))
          svg.setAttribute("height", Math.ceil(bbox.y + bbox.height + 2))
        }
      } else if (item.text) {
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
        worker.postMessage({ type: "set_theme", dark: isDark() })
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

  worker.postMessage({ type: "set_theme", dark: isDark() })
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
