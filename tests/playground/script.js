// CodeMirror (and LZString) are bundled locally into vendor/codemirror.js so
// the playground has no runtime dependency on the esm.sh CDN. Regenerate the
// bundle with `make playground-codemirror` after changing versions. See
// tests/playground-deps/ for the build sources.
import {
  EditorView, keymap, lineNumbers, highlightSpecialChars,
  highlightActiveLine,
  EditorState, Compartment,
  StreamLanguage,
  syntaxHighlighting, defaultHighlightStyle, bracketMatching,
  defaultKeymap, history, historyKeymap, indentWithTab,
  closeBrackets, closeBracketsKeymap,
  mathematica,
  oneDark,
  LZString,
} from "./vendor/codemirror.js"


let worker = null
let editorView = null

const wolframLanguage = StreamLanguage.define(mathematica)
const themeConfig = new Compartment()
// Discrete Manipulate controls with at most this many choices render as a
// segmented SetterBar (toggle buttons) whatever their labels look like; short
// labels stretch that to SETTER_BAR_MAX_COMPACT_CHOICES, and anything longer
// uses a dropdown. Mirrors rendersAsSetterBar in woxi-studio.
const SETTER_BAR_MAX_CHOICES = 4
const SETTER_BAR_MAX_COMPACT_CHOICES = 10
const SETTER_BAR_COMPACT_LABEL_CHARS = 3

// Whether a discrete control renders as a SetterBar rather than a dropdown.
// Wolfram's Manipulate decides on the choice count and the label widths (four
// phrases stay a bar; six single words become a dropdown), not on the total
// width. An icon label counts as compact — it draws at a fixed 24px.
function rendersAsSetterBar(labels, labelSvgs) {
  if (labels.length <= SETTER_BAR_MAX_CHOICES) return true
  return (
    labels.length <= SETTER_BAR_MAX_COMPACT_CHOICES &&
    labels.every(
      (label, idx) =>
        labelSvgs?.[idx] ||
        [...String(label)].length <= SETTER_BAR_COMPACT_LABEL_CHARS,
    )
  )
}
// Auto-playing widgets (Animate / ListAnimate) advance their animation
// control one step every ANIM_INTERVAL_MS. At ~60ms the default 100-step
// continuous range sweeps in ~6s, matching Wolfram's leisurely Animate.
const ANIM_INTERVAL_MS = 60
// Every running animation's stop() function, so a re-run or clear can halt
// timers that would otherwise keep firing against removed DOM nodes.
const activeAnimators = new Set()

function stopAllAnimators() {
  for (const stop of activeAnimators) stop()
  activeAnimators.clear()
}
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

function appendOutputItem(outputsEl, item) {
  if (item.type === "graphics") {
    const div = document.createElement("div")
    div.className = "output-box graphics-box"
    div.innerHTML = item.svg
    outputsEl.appendChild(div)
  } else if (item.type === "sound") {
    const div = document.createElement("div")
    div.className = "output-box sound-box"
    if (item.label) {
      const label = document.createElement("div")
      label.className = "sound-label"
      label.textContent = item.label
      div.appendChild(label)
    }
    const audio = document.createElement("audio")
    audio.controls = true
    if (item.audio) {
      audio.src =
        "data:" + (item.mime || "audio/wav") + ";base64," + item.audio
    }
    div.appendChild(audio)
    if (!item.audio) {
      // File-backed audio whose bytes are unavailable in the browser (local
      // paths cannot be read from WASM) — keep the player chrome, explain
      // why it cannot play.
      const note = document.createElement("div")
      note.className = "sound-note"
      note.textContent = "Audio file is not accessible from the browser"
      div.appendChild(note)
    }
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

function renderOutputItems(items) {
  const outputsEl = document.getElementById("outputs")
  outputsEl.innerHTML = ""
  outputsEl.classList.remove("stale")
  // Any previously pending Manipulate evaluations now target DOM
  // nodes that have been removed.  Drop their entries so stale
  // results arriving from the worker are ignored.
  manipulateRequests.clear()
  enabledRequests.clear()
  visibleRequests.clear()
  // Halt any running animation timers whose widgets are about to be replaced.
  stopAllAnimators()

  for (const item of items) {
    appendOutputItem(outputsEl, item)
  }
}

function appendOutputItems(items) {
  const outputsEl = document.getElementById("outputs")
  outputsEl.classList.remove("stale")
  for (const item of items) {
    appendOutputItem(outputsEl, item)
  }
}

// ── Manipulate interactive widget ────────────────────────────────

// Map of requestId → per-widget state object.  Used to route
// `manipulate_result` messages back to the right widget when multiple
// Manipulates coexist on one page.
const manipulateRequests = new Map()
// In-flight `evaluate_manipulate_enabled` requests, keyed by request id, so a
// result can be routed back to the widget whose controls it re-gates.
const enabledRequests = new Map()
// The same, for the requests that decide which `PaneSelector` pane's controls
// are on screen. Both use the same worker call — one evaluates `Enabled`
// conditions, the other pane conditions — so the reply carries whichever list
// its request id was registered under.
const visibleRequests = new Map()
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
    } else if (ctrl.kind === "slider2d") {
      current[ctrl.name] = { x: ctrl.xInit, y: ctrl.yInit }
    } else if (ctrl.kind === "interval") {
      current[ctrl.name] = { low: ctrl.lowInit, high: ctrl.highInit }
    } else if (ctrl.kind === "locator") {
      current[ctrl.name] = { points: ctrl.points.map((p) => [p[0], p[1]]) }
    }
  }
  // Mutable state variables (ControlType -> None) carry an InputForm value
  // string that interactive displays (e.g. a Checkbox grid) can rewrite.
  const stateVars = (item.state && typeof item.state === "object")
    ? item.state
    : {}
  for (const name in stateVars) current[name] = stateVars[name]

  // Whether this widget has extra display elements (Checkbox grids, …). When
  // it does, every re-render goes through the "full" path so the displays and
  // any checkbox write-backs stay in sync with the body.
  const hasDisplays = Array.isArray(item.displays) && item.displays.length > 0

  // Controls panel. `Appearance -> None` hides the control rows entirely
  // (the animation just runs); an animated widget keeps its play/pause bar.
  const controlsEl = document.createElement("div")
  controlsEl.className = "manipulate-controls"
  if (item.appearanceNone) controlsEl.classList.add("appearance-none")

  // Output panel (filled with the initial rendering)
  const outputEl = document.createElement("div")
  outputEl.className = "manipulate-output"
  fillManipulateOutput(outputEl, item.initial)

  // Extra-display panel (rendered from the initial widget trees, if any).
  const displaysEl = document.createElement("div")
  displaysEl.className = "manipulate-displays"

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
    // Checkbox write-backs waiting to be sent (merged while an eval is in
    // flight, then flushed on the next dispatch).
    pendingMutations: [],
    // Whether the display grid needs rebuilding on the next dispatch. Only a
    // control change (which can resize the grid) sets this; a checkbox toggle
    // leaves the grid structure intact (the clicked box already reflects its
    // new state), so we skip the expensive re-expansion and re-render only the
    // body. Cleared once a rebuild is dispatched.
    structureDirty: false,
    // Set per-dispatch: whether that request asked for fresh display trees.
    expectDisplays: false,
    // Controls with an `Enabled -> Dynamic[…]` option: `{condition, apply}`
    // in control order. `condition` is the boolean code ("" = always enabled);
    // `apply(on)` greys the control's DOM in or out. Re-evaluated on every
    // binding change so a control can disable itself for the current state.
    enabledControls: [],
    // Controls belonging to a `PaneSelector` pane: `{condition, apply}` in
    // control order, where `condition` decides whether that pane is the one
    // on screen and `apply(on)` shows or hides the control's row. Controls
    // outside any pane are not listed and are always shown.
    visibleControls: [],
    // Continuous-slider drivers an Animate/ListAnimate animator can advance.
    animatables: [],
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
      // Annotation rows (headings / delimiters) bind no variable.
      if (ctrl.kind === "heading" || ctrl.kind === "delimiter") continue
      const v = current[ctrl.name]
      if (ctrl.kind === "slider2d") {
        bindings[ctrl.name] = `{${v.x}, ${v.y}}`
      } else if (ctrl.kind === "interval") {
        bindings[ctrl.name] = `{${v.low}, ${v.high}}`
      } else if (ctrl.kind === "locator") {
        // Locator positions are machine reals: integral coordinates keep a
        // trailing dot so they substitute as Real, not Integer.
        const real = (n) => (Number.isInteger(n) ? `${n}.` : String(n))
        bindings[ctrl.name] =
          `{${v.points.map(([x, y]) => `{${real(x)}, ${real(y)}}`).join(", ")}}`
      } else {
        bindings[ctrl.name] = String(v)
      }
    }
    // State variables already hold InputForm value strings.
    for (const name in stateVars) bindings[name] = current[name]
    return bindings
  }

  // `mutations` is an optional array of write-back assignments (e.g.
  // `data[[3, 5]] = 1`) produced by toggling a checkbox. A control change
  // (no mutations) can resize the grid, so it marks the structure dirty; a
  // checkbox toggle does not (only the body needs re-rendering).
  function requestUpdate(mutations) {
    if (!worker) return
    if (Array.isArray(mutations) && mutations.length > 0) {
      widget.pendingMutations.push(...mutations)
    } else {
      widget.structureDirty = true
    }
    const bindings = buildBindings()
    if (widget.inflight) {
      // An evaluation is already running; remember the latest bindings.
      // Pending mutations accumulate and are flushed on the next dispatch.
      widget.pendingBindings = bindings
      return
    }
    dispatchUpdate(bindings)
  }

  // Re-evaluate a list of gated controls' conditions for the current bindings
  // and apply the resulting flags. A no-op when no control has a condition.
  function refreshGated(controls, requests, bindings) {
    if (!worker) return
    const conditions = controls.map((c) => c.condition)
    if (conditions.every((c) => !c)) return
    const requestId = ++manipulateRequestCounter
    requests.set(requestId, widget)
    worker.postMessage({
      type: "evaluate_manipulate_enabled",
      requestId,
      conditions,
      bindings: bindings || buildBindings(),
    })
  }

  // Grey the controls an `Enabled -> Dynamic[…]` option has switched off in
  // or out; show only the controls of the `PaneSelector` pane in force.
  function refreshEnabled(bindings) {
    refreshGated(widget.enabledControls, enabledRequests, bindings)
    refreshGated(widget.visibleControls, visibleRequests, bindings)
  }
  widget.refreshEnabled = refreshEnabled

  function dispatchUpdate(bindings) {
    widget.inflight = true
    widget.pendingBindings = null
    refreshEnabled(bindings)
    const requestId = ++manipulateRequestCounter
    manipulateRequests.set(requestId, widget)
    outputEl.classList.add("stale")
    if (hasDisplays) {
      const mutations = widget.pendingMutations
      widget.pendingMutations = []
      // Only rebuild the (expensive) display grid when its structure may have
      // changed. A pure checkbox toggle re-renders just the body.
      const wantDisplays = widget.structureDirty
      widget.structureDirty = false
      widget.expectDisplays = wantDisplays
      worker.postMessage({
        type: "evaluate_manipulate_full",
        requestId,
        body: item.body,
        displays: wantDisplays ? item.displays : [],
        bindings,
        mutations,
      })
    } else {
      worker.postMessage({
        type: "evaluate_manipulate",
        requestId,
        body: item.body,
        bindings,
      })
    }
  }

  widget.dispatchUpdate = dispatchUpdate

  // Render the widget trees for the extra display elements. Each checkbox
  // toggle produces a `target = value` write-back that re-renders the widget.
  function renderDisplayTree(node) {
    if (!node || typeof node !== "object") return document.createTextNode("")
    switch (node.kind) {
      case "panel": {
        const div = document.createElement("div")
        div.className = "manipulate-display-panel"
        div.appendChild(renderDisplayTree(node.child))
        return div
      }
      case "grid": {
        const grid = document.createElement("div")
        grid.className = "manipulate-display-grid"
        for (const row of node.rows || []) {
          const tr = document.createElement("div")
          tr.className = "manipulate-display-grid-row"
          for (const cell of row) tr.appendChild(renderDisplayTree(cell))
          grid.appendChild(tr)
        }
        return grid
      }
      case "column":
      case "row": {
        const div = document.createElement("div")
        div.className = node.kind === "row"
          ? "manipulate-display-row"
          : "manipulate-display-column"
        for (const c of node.children || []) {
          div.appendChild(renderDisplayTree(c))
        }
        return div
      }
      case "checkbox": {
        const input = document.createElement("input")
        input.type = "checkbox"
        input.checked = !!node.checked
        input.className = "manipulate-display-checkbox"
        if (node.target) {
          input.addEventListener("change", () => {
            const val = input.checked ? node.on : node.off
            requestUpdate([`${node.target} = ${val}`])
          })
        } else {
          input.disabled = true
        }
        return input
      }
      case "toggler": {
        // One choice of a TogglerBar[Dynamic[var], …]: clicking adds or
        // removes its value from the bound list variable.
        const btn = document.createElement("button")
        btn.type = "button"
        btn.className = node.selected
          ? "manipulate-display-toggler selected"
          : "manipulate-display-toggler"
        if (node.label) btn.appendChild(renderDisplayTree(node.label))
        if (node.mutation) {
          btn.addEventListener("click", () => {
            requestUpdate([node.mutation])
          })
        }
        return btn
      }
      case "button": {
        // A caption button (`Button["→", n++]`): its held action is sent as
        // a write-back and the caption is rebuilt so it shows the new value.
        const btn = document.createElement("button")
        btn.type = "button"
        btn.className = "manipulate-display-button"
        if (node.label) btn.appendChild(renderDisplayTree(node.label))
        if (node.action) {
          btn.addEventListener("click", () => {
            widget.structureDirty = true
            requestUpdate([node.action])
          })
        }
        return btn
      }
      case "spacer": {
        const span = document.createElement("span")
        span.className = "manipulate-display-spacer"
        span.style.display = "inline-block"
        span.style.width = `${node.width || 0}px`
        return span
      }
      case "text": {
        // Styled prose (`Style["…", Bold, Red]`) inside a caption.
        const span = document.createElement("span")
        span.className = "manipulate-display-text"
        for (const run of node.runs || []) {
          const s = document.createElement("span")
          s.textContent = run.text
          if (run.italic) s.style.fontStyle = "italic"
          if (run.bold) s.style.fontWeight = "bold"
          if (run.color) s.style.color = run.color
          span.appendChild(s)
        }
        return span
      }
      case "static": {
        const div = document.createElement("div")
        if (node.svg) {
          div.className = "graphics-box"
          div.innerHTML = node.svg
        } else {
          div.className = "text-box"
          div.textContent = node.text || ""
        }
        return div
      }
      default:
        return document.createTextNode("")
    }
  }

  function renderDisplays(trees) {
    displaysEl.innerHTML = ""
    if (!Array.isArray(trees)) return
    for (const tree of trees) displaysEl.appendChild(renderDisplayTree(tree))
  }
  widget.renderDisplays = renderDisplays

  // Populate the initial display trees, if the widget shipped any.
  if (hasDisplays && item.initial && item.initial.displayTrees) {
    renderDisplays(item.initial.displayTrees)
  }

  for (const ctrl of item.controls) {
    // Annotation rows between controls: a heading (a string or `Style[…]`
    // Manipulate argument, e.g. "signal 1") or a `Delimiter` separator.
    // They bind no variable and span the full row.
    if (ctrl.kind === "heading") {
      const heading = document.createElement("div")
      heading.className = "manipulate-heading"
      const runs = ctrl.labelRuns
      if (Array.isArray(runs) && runs.length > 0) {
        for (const run of runs) {
          const part = document.createElement("span")
          part.textContent = run.text
          if (run.italic) part.style.fontStyle = "italic"
          heading.appendChild(part)
        }
      } else {
        heading.textContent = ctrl.label || ""
      }
      controlsEl.appendChild(heading)
      continue
    }
    if (ctrl.kind === "delimiter") {
      const hr = document.createElement("hr")
      hr.className = "manipulate-delimiter"
      controlsEl.appendChild(hr)
      continue
    }

    const row = document.createElement("label")
    row.className = "manipulate-control-row"

    const lbl = document.createElement("span")
    lbl.className = "manipulate-label"
    // Render the label's styled runs (italic where flagged, e.g. an italic
    // `t` or the italic `m` of `m₁`). Fall back to the plain label / name
    // when no runs are present.
    const runs = ctrl.labelRuns
    if (Array.isArray(runs) && runs.length > 0) {
      for (const run of runs) {
        const part = document.createElement("span")
        part.textContent = run.text
        if (run.italic) part.style.fontStyle = "italic"
        lbl.appendChild(part)
      }
    } else {
      lbl.textContent = ctrl.label || ctrl.name
    }
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
      widget.enabledControls.push({
        condition: ctrl.enabledWhen || "",
        apply: (on) => {
          input.disabled = !on
          row.classList.toggle("disabled", !on)
        },
      })
      // Expose this slider so an Animate/ListAnimate animator can drive it.
      widget.animatables.push({
        min: Number(ctrl.min),
        max: Number(ctrl.max),
        step: step > 0 ? Number(step) : (Number(ctrl.max) - Number(ctrl.min)) / 100,
        get: () => Number(input.value),
        set: (v) => {
          input.value = v
          current[ctrl.name] = String(v)
          display.textContent = fmt(v)
          requestUpdate()
        },
      })
    } else if (ctrl.kind === "discrete") {
      // The bound value is `values[idx]`; the visible text is the parallel
      // `valueLabels[idx]` (a rule's right-hand side for a `value -> "label"`
      // choice, else the value itself).
      const labels = ctrl.valueLabels ?? ctrl.values
      const values = ctrl.values
      // `ControlType -> PopupMenu` always renders a dropdown, matching
      // Wolfram; a small anonymous set defaults to a SetterBar.
      if (rendersAsSetterBar(labels, ctrl.valueLabelSvgs) && !ctrl.popup) {
        // A small enumerated set renders as a segmented SetterBar: a row of
        // adjacent toggle buttons with the active choice highlighted, matching
        // Wolfram's SetterBar.
        const bar = document.createElement("div")
        bar.className = "manipulate-setterbar"
        const buttons = []
        values.forEach((val, idx) => {
          const btn = document.createElement("button")
          btn.type = "button"
          btn.className = "setter-btn"
          // A choice whose rule label is a graphic (`"+" -> myIcon[2]`)
          // shows the rendered SVG icon; text choices show their label.
          const iconSvg = ctrl.valueLabelSvgs?.[idx]
          if (iconSvg) {
            btn.innerHTML = iconSvg
            btn.classList.add("setter-btn-icon")
          } else {
            btn.textContent = labels[idx] ?? val
          }
          if (idx === ctrl.initialIndex) btn.classList.add("active")
          btn.addEventListener("click", () => {
            current[ctrl.name] = val
            for (const b of buttons) b.classList.remove("active")
            btn.classList.add("active")
            requestUpdate()
          })
          buttons.push(btn)
          bar.appendChild(btn)
        })
        row.appendChild(bar)
        widget.enabledControls.push({
          condition: ctrl.enabledWhen || "",
          apply: (on) => {
            for (const b of buttons) b.disabled = !on
            bar.classList.toggle("disabled", !on)
            row.classList.toggle("disabled", !on)
          },
        })
      } else {
        // Larger sets fall back to a dropdown so the row can't grow unbounded.
        const select = document.createElement("select")
        for (let idx = 0; idx < values.length; idx++) {
          const opt = document.createElement("option")
          opt.value = values[idx]
          opt.textContent = labels[idx] ?? values[idx]
          if (idx === ctrl.initialIndex) opt.selected = true
          select.appendChild(opt)
        }
        row.appendChild(select)
        select.addEventListener("change", () => {
          current[ctrl.name] = select.value
          requestUpdate()
        })
        widget.enabledControls.push({
          condition: ctrl.enabledWhen || "",
          apply: (on) => {
            select.disabled = !on
            row.classList.toggle("disabled", !on)
          },
        })
      }

      // The controls share one grid (rows use `display: contents`), so every
      // row must contribute the same three cells or auto-placement desyncs the
      // shared label column. Discrete controls have no value readout, so add an
      // empty cell to fill the third column.
      const spacer = document.createElement("span")
      spacer.className = "manipulate-value"
      row.appendChild(spacer)
    } else if (ctrl.kind === "slider2d") {
      // A 2D draggable pad. The handle position maps linearly onto the
      // [xMin,xMax] × [yMin,yMax] range; the bound value is `{x, y}`.
      const pad = document.createElement("div")
      pad.className = "manipulate-pad"
      const handle = document.createElement("div")
      handle.className = "manipulate-pad-handle"
      pad.appendChild(handle)

      const display = document.createElement("span")
      display.className = "manipulate-value"

      const xSpan = ctrl.xMax - ctrl.xMin
      const ySpan = ctrl.yMax - ctrl.yMin

      function placeHandle(x, y) {
        const fx = xSpan !== 0 ? (x - ctrl.xMin) / xSpan : 0.5
        const fy = ySpan !== 0 ? (y - ctrl.yMin) / ySpan : 0.5
        handle.style.left = `${fx * 100}%`
        handle.style.bottom = `${fy * 100}%`
        display.textContent = `{${fmt(x)}, ${fmt(y)}}`
      }
      placeHandle(current[ctrl.name].x, current[ctrl.name].y)

      let dragging = false
      function updateFromPointer(ev) {
        const rect = pad.getBoundingClientRect()
        let fx = rect.width !== 0 ? (ev.clientX - rect.left) / rect.width : 0
        let fy = rect.height !== 0 ? 1 - (ev.clientY - rect.top) / rect.height : 0
        fx = Math.max(0, Math.min(1, fx))
        fy = Math.max(0, Math.min(1, fy))
        const x = ctrl.xMin + fx * xSpan
        const y = ctrl.yMin + fy * ySpan
        current[ctrl.name] = { x, y }
        placeHandle(x, y)
        requestUpdate()
      }
      pad.addEventListener("pointerdown", (ev) => {
        dragging = true
        pad.setPointerCapture(ev.pointerId)
        updateFromPointer(ev)
        ev.preventDefault()
      })
      pad.addEventListener("pointermove", (ev) => {
        if (dragging) updateFromPointer(ev)
      })
      pad.addEventListener("pointerup", () => {
        dragging = false
      })

      row.appendChild(pad)
      row.appendChild(display)
      widget.enabledControls.push({
        condition: ctrl.enabledWhen || "",
        apply: (on) => {
          pad.style.pointerEvents = on ? "" : "none"
          row.classList.toggle("disabled", !on)
        },
      })
    } else if (ctrl.kind === "locator") {
      // A list of draggable points on a 2D pad (e.g. polygon vertices).
      // Dragging moves the nearest handle. With `LocatorAutoCreate`,
      // clicking empty pad space adds a point there and double-clicking a
      // handle removes it, mirroring Wolfram's locators.
      const pad = document.createElement("div")
      pad.className = "manipulate-pad manipulate-locator-pad"
      const display = document.createElement("span")
      display.className = "manipulate-value"

      const xSpan = ctrl.xMax - ctrl.xMin
      const ySpan = ctrl.yMax - ctrl.yMin
      let handles = []

      function refreshDisplay() {
        const pts = current[ctrl.name].points
        display.textContent =
          pts.map(([x, y]) => `{${fmt(x)}, ${fmt(y)}}`).join(" ")
      }
      function rebuildHandles() {
        for (const h of handles) h.remove()
        handles = []
        current[ctrl.name].points.forEach((pt, idx) => {
          const handle = document.createElement("div")
          handle.className = "manipulate-pad-handle"
          const fx = xSpan !== 0 ? (pt[0] - ctrl.xMin) / xSpan : 0.5
          const fy = ySpan !== 0 ? (pt[1] - ctrl.yMin) / ySpan : 0.5
          handle.style.left = `${fx * 100}%`
          handle.style.bottom = `${fy * 100}%`
          if (ctrl.autoCreate) {
            handle.addEventListener("dblclick", () => {
              current[ctrl.name].points.splice(idx, 1)
              rebuildHandles()
              refreshDisplay()
              requestUpdate()
            })
          }
          pad.appendChild(handle)
          handles.push(handle)
        })
      }
      rebuildHandles()
      refreshDisplay()

      function padCoords(ev) {
        const rect = pad.getBoundingClientRect()
        let fx = rect.width !== 0 ? (ev.clientX - rect.left) / rect.width : 0
        let fy =
          rect.height !== 0 ? 1 - (ev.clientY - rect.top) / rect.height : 0
        fx = Math.max(0, Math.min(1, fx))
        fy = Math.max(0, Math.min(1, fy))
        return [ctrl.xMin + fx * xSpan, ctrl.yMin + fy * ySpan]
      }
      let dragIndex = -1
      pad.addEventListener("pointerdown", (ev) => {
        const [x, y] = padCoords(ev)
        const pts = current[ctrl.name].points
        // Pick the handle nearest the pointer (within 8% of the pad).
        let best = -1
        let bestD = Infinity
        pts.forEach(([px, py], i) => {
          const d = Math.hypot(
            xSpan !== 0 ? (px - x) / xSpan : 0,
            ySpan !== 0 ? (py - y) / ySpan : 0,
          )
          if (d < bestD) {
            bestD = d
            best = i
          }
        })
        if (best >= 0 && bestD <= 0.08) {
          dragIndex = best
        } else if (ctrl.autoCreate) {
          pts.push([x, y])
          dragIndex = pts.length - 1
          rebuildHandles()
          refreshDisplay()
          requestUpdate()
        } else {
          dragIndex = best
        }
        if (dragIndex >= 0) pad.setPointerCapture(ev.pointerId)
        ev.preventDefault()
      })
      pad.addEventListener("pointermove", (ev) => {
        if (dragIndex < 0) return
        const [x, y] = padCoords(ev)
        current[ctrl.name].points[dragIndex] = [x, y]
        const handle = handles[dragIndex]
        if (handle) {
          const fx = xSpan !== 0 ? (x - ctrl.xMin) / xSpan : 0.5
          const fy = ySpan !== 0 ? (y - ctrl.yMin) / ySpan : 0.5
          handle.style.left = `${fx * 100}%`
          handle.style.bottom = `${fy * 100}%`
        }
        refreshDisplay()
        requestUpdate()
      })
      pad.addEventListener("pointerup", () => {
        dragIndex = -1
      })

      row.appendChild(pad)
      row.appendChild(display)
      widget.enabledControls.push({
        condition: ctrl.enabledWhen || "",
        apply: (on) => {
          pad.style.pointerEvents = on ? "" : "none"
          row.classList.toggle("disabled", !on)
        },
      })
    } else if (ctrl.kind === "interval") {
      // Two range inputs (low and high endpoints) kept ordered so the
      // bound value `{low, high}` is always a valid interval.
      const stack = document.createElement("div")
      stack.className = "manipulate-interval"

      const step = ctrl.step ?? (ctrl.max - ctrl.min) / 100
      const lowInput = document.createElement("input")
      lowInput.type = "range"
      lowInput.min = ctrl.min
      lowInput.max = ctrl.max
      lowInput.step = step > 0 ? step : "any"
      lowInput.value = ctrl.lowInit
      const highInput = document.createElement("input")
      highInput.type = "range"
      highInput.min = ctrl.min
      highInput.max = ctrl.max
      highInput.step = step > 0 ? step : "any"
      highInput.value = ctrl.highInit
      stack.appendChild(lowInput)
      stack.appendChild(highInput)

      const display = document.createElement("span")
      display.className = "manipulate-value"
      display.textContent = `{${fmt(ctrl.lowInit)}, ${fmt(ctrl.highInit)}}`

      function syncInterval() {
        let low = Number(lowInput.value)
        let high = Number(highInput.value)
        if (low > high) {
          // Whichever thumb crossed over drags the other with it.
          [low, high] = [Math.min(low, high), Math.max(low, high)]
          lowInput.value = low
          highInput.value = high
        }
        current[ctrl.name] = { low, high }
        display.textContent = `{${fmt(low)}, ${fmt(high)}}`
        requestUpdate()
      }
      lowInput.addEventListener("input", syncInterval)
      highInput.addEventListener("input", syncInterval)

      row.appendChild(stack)
      row.appendChild(display)
      widget.enabledControls.push({
        condition: ctrl.enabledWhen || "",
        apply: (on) => {
          lowInput.disabled = !on
          highInput.disabled = !on
          row.classList.toggle("disabled", !on)
        },
      })
    }

    // A control that came from a `PaneSelector` pane is only on screen while
    // the selector holds that pane's value. Every control kind builds a `row`,
    // so the pane gate is registered once here rather than per kind.
    if (ctrl.visibleWhen) {
      widget.visibleControls.push({
        condition: ctrl.visibleWhen,
        apply: (on) => {
          // Rows are `display: contents` in the control grid, so clearing the
          // inline style is what puts one back rather than setting a value.
          row.style.display = on ? "" : "none"
        },
      })
    }
    controlsEl.appendChild(row)
  }

  // Grey out any control that starts disabled for the initial bindings.
  widget.refreshEnabled(buildBindings())

  // An Animate/ListAnimate widget auto-plays: a play/pause button advances its
  // first continuous slider on a timer, looping min → max → min.
  if (item.animated && widget.animatables.length > 0) {
    const target = widget.animatables[0]
    const bar = document.createElement("div")
    bar.className = "manipulate-animator"

    const btn = document.createElement("button")
    btn.type = "button"
    btn.className = "manipulate-play"

    let timer = null
    function stop() {
      if (timer !== null) {
        clearInterval(timer)
        timer = null
      }
      btn.textContent = "▶"
      btn.setAttribute("aria-label", "Play")
      activeAnimators.delete(stop)
    }
    function tick() {
      let v = target.get() + target.step
      // Loop back to the start once we step past the end (small epsilon so
      // floating-point drift doesn't skip the final frame).
      if (v > target.max + target.step * 1e-6) v = target.min
      target.set(v)
    }
    function play() {
      if (timer !== null) return
      btn.textContent = "❚❚"
      btn.setAttribute("aria-label", "Pause")
      timer = setInterval(tick, ANIM_INTERVAL_MS)
      activeAnimators.add(stop)
    }
    btn.addEventListener("click", () => {
      if (timer === null) play()
      else stop()
    })
    widget.stopAnimation = stop

    bar.appendChild(btn)
    controlsEl.appendChild(bar)
    // Start playing immediately (Wolfram's default AnimationRunning -> True)
    // unless the widget was built paused (`AnimationRunning -> False`).
    if (item.animationRunning !== false) play()
  }

  box.appendChild(controlsEl)
  // Extra display elements (e.g. the Checkbox grid) sit above the rendered
  // body output.
  if (hasDisplays) box.appendChild(displaysEl)
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
  stopAllAnimators()
  document.getElementById("outputs").innerHTML = ""
}

function initWorker() {
  showStatus("Loading Woxi WebAssembly module ...", "info", { autoHide: false })

  // Cache-bust the worker script so a stale worker.js (e.g. one missing a
  // newly added message handler) can never be served against a fresh wasm
  // build — that manifests as messages being silently dropped (e.g. a
  // Manipulate slider that no longer updates the graphic).
  worker = new Worker("worker.js?v=" + Date.now(), { type: "module" })

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
    else if (type === "partial_result") {
      if (success) {
        try {
          const items = JSON.parse(result)
          appendOutputItems(items)
        } catch (_) {
          appendOutputItems([{ type: "error", text: result }])
        }
      }
    }
    else if (type === "result_done") {
      hideSpinner("runSpinner")
      document.getElementById("runBtn").disabled = false
      if (!success) {
        appendOutputItems([{ type: "error", text: message }])
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
    else if (type === "manipulate_enabled_result") {
      // The request id says which gate this reply answers: the `Enabled`
      // conditions or the `PaneSelector` pane conditions.
      const enabledWidget = enabledRequests.get(e.data.requestId)
      const visibleWidget = visibleRequests.get(e.data.requestId)
      enabledRequests.delete(e.data.requestId)
      visibleRequests.delete(e.data.requestId)
      const widget = enabledWidget || visibleWidget
      if (!widget || !success) return
      let flags
      try {
        flags = JSON.parse(result)
      } catch (_) {
        return
      }
      if (!Array.isArray(flags)) return
      const gated = enabledWidget
        ? widget.enabledControls
        : widget.visibleControls
      gated.forEach((c, i) => {
        // Absent/undefined flag fails open (enabled / on screen).
        c.apply(flags[i] !== false)
      })
    }
    else if (type === "manipulate_full_result") {
      const widget = manipulateRequests.get(e.data.requestId)
      manipulateRequests.delete(e.data.requestId)
      if (!widget) return
      widget.inflight = false
      if (success) {
        try {
          const payload = JSON.parse(result)
          fillManipulateOutput(widget.outputEl, payload.output || {})
          // Only replace the display DOM when this request asked for a
          // rebuild; a checkbox toggle leaves the existing grid in place (the
          // clicked box already shows its new state).
          if (widget.expectDisplays) widget.renderDisplays(payload.displays)
          // Fold any updated state values back into the binding set so the
          // next toggle/slider builds on the current matrix.
          if (payload.state && typeof payload.state === "object") {
            for (const name in payload.state) {
              widget.current[name] = payload.state[name]
            }
          }
        } catch (_) {
          fillManipulateOutput(widget.outputEl, { error: result })
        }
      } else {
        fillManipulateOutput(widget.outputEl, { error: message })
      }
      // Flush any binding/mutation update queued while we were busy. Queued
      // mutations always travel with `pendingBindings` (set together in
      // requestUpdate), so dispatching on the latter also flushes them.
      if (widget.pendingBindings) {
        const next = widget.pendingBindings
        widget.pendingBindings = null
        widget.dispatchUpdate(next)
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
  clearOutputs()
  // Drop any pending manipulate requests targeting now-removed DOM nodes.
  manipulateRequests.clear()
  document.getElementById("outputs").classList.remove("stale")

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
