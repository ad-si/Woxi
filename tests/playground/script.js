import {
  EditorView, keymap, lineNumbers, highlightSpecialChars,
  highlightActiveLine,
} from "https://esm.sh/@codemirror/view@6"
import { EditorState, Compartment } from "https://esm.sh/@codemirror/state@6"
import {
  StreamLanguage,
  syntaxHighlighting, defaultHighlightStyle, bracketMatching,
} from "https://esm.sh/@codemirror/language@6"
import {
  defaultKeymap, history, historyKeymap, indentWithTab,
} from "https://esm.sh/@codemirror/commands@6"
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
const DEFAULT_CODE = `GraphicsRow[{
  Plot[Sin[x], {x, 0, 4 Pi}],
  BarChart[{1, 5, 3, 4, 7, 9}],
  NumberLinePlot[{Interval[{1, 9}], Interval[{3, 7}], Interval[{2, 4}]}]
}]
GraphicsRow[{
  BubbleChart[{{1, 5, 3}, {4, 6, 9}}],
  Graph[{a -> b, b -> a, b <-> b}],
  TreeForm[a + b^2 + c^3 + d]
}]`

function saveEditorContent() {
  const content = editorView.state.doc.toString()
  localStorage.setItem(STORAGE_KEY, content)
}

const persistenceListener = EditorView.updateListener.of((update) => {
  if (update.docChanged) {
    saveEditorContent()
    document.getElementById("outputs").classList.add("stale")
  }
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
      indentWithTab,
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


let statusHideTimer = null

function showStatus(message, type = "info", { autoHide = true } = {}) {
  const status = document.getElementById("status")

  if (statusHideTimer) {
    clearTimeout(statusHideTimer)
    statusHideTimer = null
  }

  if (message === "") {
    status.style.display = "none"
    return
  }

  status.textContent = message
  status.className = `status ${type}`
  status.style.display = "block"

  if (type === "info" && autoHide) {
    statusHideTimer = setTimeout(() => {
      status.style.display = "none"
      statusHideTimer = null
    }, 3000)
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
  outputsEl.classList.remove("stale")
  // Any previously pending Manipulate evaluations now target DOM
  // nodes that have been removed.  Drop their entries so stale
  // results arriving from the worker are ignored.
  manipulateRequests.clear()

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
      } else if (item.text) {
        const pre = document.createElement("pre")
        pre.className = "output-box text-box"
        pre.textContent = item.text
        outputsEl.appendChild(pre)
      }
    } else if (item.type === "manipulate") {
      outputsEl.appendChild(renderManipulate(item))
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

// ── Manipulate interactive widget ────────────────────────────────

// Map of requestId → per-widget state object.  Used to route
// `manipulate_result` messages back to the right widget when multiple
// Manipulates coexist on one page.
const manipulateRequests = new Map()
let manipulateRequestCounter = 0

function renderManipulate(item) {
  const box = document.createElement("div")
  box.className = "output-box manipulate-box"

  // Current values for each control, keyed by variable name.
  const current = {}
  for (const ctrl of item.controls) {
    if (ctrl.kind === "continuous") {
      current[ctrl.name] = ctrl.initial
    } else if (ctrl.kind === "discrete") {
      current[ctrl.name] = ctrl.values[ctrl.initialIndex] ?? ctrl.values[0]
    }
  }

  // Controls panel
  const controlsEl = document.createElement("div")
  controlsEl.className = "manipulate-controls"

  // Output panel (filled with the initial rendering)
  const outputEl = document.createElement("div")
  outputEl.className = "manipulate-output"
  fillManipulateOutput(outputEl, item.initial)

  // Per-widget coalescing state.  At most one evaluation is in flight
  // at a time: slider events that fire while an eval is pending only
  // update `pendingBindings`, and the next eval is dispatched when the
  // previous one completes.  This prevents a queue of stale frames
  // from building up while the user drags a slider.
  const widget = {
    item,
    outputEl,
    current,
    inflight: false,
    pendingBindings: null,
  }

  // Format a continuous value for display (strip trailing zeros).
  function fmt(v) {
    const n = Number(v)
    if (!Number.isFinite(n)) return String(v)
    return Number.isInteger(n) ? String(n) : n.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")
  }

  function buildBindings() {
    const bindings = {}
    for (const ctrl of item.controls) {
      bindings[ctrl.name] = String(current[ctrl.name])
    }
    return bindings
  }

  function requestUpdate() {
    if (!worker) return
    const bindings = buildBindings()
    if (widget.inflight) {
      // An evaluation is already running; just remember the latest
      // desired bindings.  Whatever was previously pending is dropped.
      widget.pendingBindings = bindings
      return
    }
    dispatchUpdate(bindings)
  }

  function dispatchUpdate(bindings) {
    widget.inflight = true
    widget.pendingBindings = null
    const requestId = ++manipulateRequestCounter
    manipulateRequests.set(requestId, widget)
    outputEl.classList.add("stale")
    worker.postMessage({
      type: "evaluate_manipulate",
      requestId,
      body: item.body,
      bindings,
    })
  }

  widget.dispatchUpdate = dispatchUpdate

  for (const ctrl of item.controls) {
    const row = document.createElement("label")
    row.className = "manipulate-control-row"

    const lbl = document.createElement("span")
    lbl.className = "manipulate-label"
    lbl.textContent = ctrl.label || ctrl.name
    row.appendChild(lbl)

    if (ctrl.kind === "continuous") {
      const input = document.createElement("input")
      input.type = "range"
      input.min = ctrl.min
      input.max = ctrl.max
      const step = ctrl.step ?? (ctrl.max - ctrl.min) / 100
      input.step = step > 0 ? step : "any"
      input.value = ctrl.initial
      row.appendChild(input)

      const display = document.createElement("span")
      display.className = "manipulate-value"
      display.textContent = fmt(ctrl.initial)
      row.appendChild(display)

      input.addEventListener("input", () => {
        current[ctrl.name] = input.value
        display.textContent = fmt(input.value)
        requestUpdate()
      })
    } else if (ctrl.kind === "discrete") {
      const select = document.createElement("select")
      for (let idx = 0; idx < ctrl.values.length; idx++) {
        const opt = document.createElement("option")
        opt.value = ctrl.values[idx]
        opt.textContent = ctrl.values[idx]
        if (idx === ctrl.initialIndex) opt.selected = true
        select.appendChild(opt)
      }
      row.appendChild(select)
      select.addEventListener("change", () => {
        current[ctrl.name] = select.value
        requestUpdate()
      })
    }

    controlsEl.appendChild(row)
  }

  box.appendChild(controlsEl)
  box.appendChild(outputEl)
  return box
}

function fillManipulateOutput(el, payload) {
  el.classList.remove("stale")
  el.innerHTML = ""
  if (!payload) return
  if (payload.error) {
    const pre = document.createElement("pre")
    pre.className = "error-box"
    pre.textContent = payload.error
    el.appendChild(pre)
    return
  }
  if (payload.svg) {
    const div = document.createElement("div")
    div.className = "graphics-box"
    div.innerHTML = payload.svg
    el.appendChild(div)
    return
  }
  if (payload.textSvg) {
    const div = document.createElement("div")
    div.className = "text-box"
    div.innerHTML = payload.textSvg
    el.appendChild(div)
    return
  }
  if (payload.text) {
    const pre = document.createElement("pre")
    pre.className = "text-box"
    pre.textContent = payload.text
    el.appendChild(pre)
  }
}

function clearOutputs() {
  document.getElementById("outputs").innerHTML = ""
}

function initWorker() {
  showStatus("Loading Woxi WebAssembly module ...", "info", { autoHide: false })

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
    else if (type === "manipulate_result") {
      const widget = manipulateRequests.get(e.data.requestId)
      manipulateRequests.delete(e.data.requestId)
      if (!widget) return
      widget.inflight = false
      if (success) {
        try {
          const payload = JSON.parse(result)
          fillManipulateOutput(widget.outputEl, payload)
        } catch (_) {
          fillManipulateOutput(widget.outputEl, { error: result })
        }
      } else {
        fillManipulateOutput(widget.outputEl, { error: message })
      }
      // Dispatch any pending update that arrived while we were busy.
      // This keeps the pipeline at depth 1: at most one eval running
      // plus one most-recent binding queued.
      if (widget.pendingBindings) {
        const next = widget.pendingBindings
        widget.pendingBindings = null
        widget.dispatchUpdate(next)
      }
      // Note: we intentionally skip `saveOutput()` on manipulate_result
      // so rapid slider movement doesn't trigger repeated localStorage
      // writes of full SVGs.  The initial rendering is already
      // persisted when the "result" message first arrived.
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
