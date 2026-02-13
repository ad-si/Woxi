import { Marked } from "https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js"
import DOMPurify from "https://cdn.jsdelivr.net/npm/dompurify/dist/purify.es.mjs"

const mathInline = {
  name: "mathInline",
  level: "inline",
  start(src) { return src.indexOf("$") },
  tokenizer(src) {
    const match = src.match(/^\$([^\$\n]+?)\$/)
    if (match) {
      return { type: "mathInline", raw: match[0], text: match[1].trim() }
    }
  },
  renderer(token) {
    try {
      return window.katex.renderToString(token.text, { throwOnError: false })
    } catch {
      return `<code>${escapeHtml(token.text)}</code>`
    }
  },
}

const mathBlock = {
  name: "mathBlock",
  level: "block",
  start(src) { return src.indexOf("$$") },
  tokenizer(src) {
    const match = src.match(/^\$\$([\s\S]+?)\$\$/)
    if (match) {
      return { type: "mathBlock", raw: match[0], text: match[1].trim() }
    }
  },
  renderer(token) {
    try {
      return `<div class="katex-display">${window.katex.renderToString(token.text, { throwOnError: false, displayMode: true })}</div>`
    } catch {
      return `<pre><code>${escapeHtml(token.text)}</code></pre>`
    }
  },
}

const marked = new Marked({
  breaks: true,
  gfm: true,
  extensions: [mathBlock, mathInline],
})

function highlightCode(code, lang) {
  if (lang && window.hljs && window.hljs.getLanguage(lang)) {
    return window.hljs.highlight(code, { language: lang }).value
  }
  if (window.hljs) {
    return window.hljs.highlightAuto(code).value
  }
  return escapeHtml(code)
}

marked.setOptions({
  highlight: highlightCode,
})

// Override the code renderer to use hljs
const renderer = new marked.Renderer()
renderer.code = function ({ text, lang }) {
  const highlighted = highlightCode(text, lang)
  return `<pre><code class="hljs${lang ? ` language-${lang}` : ''}">${highlighted}</code></pre>`
}
marked.use({ renderer })

function escapeHtml(text) {
  const div = document.createElement("div")
  div.textContent = text
  return div.innerHTML
}

export function renderMarkdown(text) {
  const html = marked.parse(text)
  return DOMPurify.sanitize(html)
}

export function createMessageElement(msg, index) {
  if (msg.role === "user") return createUserMessage(msg, index)
  if (msg.role === "assistant") return createAssistantMessage(msg)
  if (msg.role === "tool") return createToolResultMessage(msg)
  return null
}

function createUserMessage(msg, index) {
  const el = document.createElement("div")
  el.className = "user-message-row flex flex-col items-end px-4 py-3"
  if (index !== undefined) el.dataset.msgIndex = index
  el.innerHTML = `
    <div class="max-w-[80%] px-4 py-2.5 rounded-2xl bg-blue-600 text-white text-sm whitespace-pre-wrap">${escapeHtml(msg.content)}</div>
    <div class="user-msg-actions">
      <button class="msg-action-btn" data-action="copy" title="Copy">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" stroke-width="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" stroke-width="2"/></svg>
      </button>
      <button class="msg-action-btn" data-action="edit" title="Edit">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
      </button>
      <button class="msg-action-btn" data-action="retry" title="Retry">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M1 4v6h6"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>
      </button>
    </div>
  `
  return el
}

function createAssistantMessage(msg) {
  const wrapper = document.createElement("div")
  wrapper.className = "px-4 py-3"

  const inner = document.createElement("div")
  inner.className = "max-w-3xl mx-auto"

  // Text content
  if (msg.content) {
    const contentEl = document.createElement("div")
    contentEl.className = "message-content text-sm"
    contentEl.innerHTML = renderMarkdown(msg.content)
    inner.appendChild(contentEl)
  }

  // Tool calls
  if (msg.tool_calls) {
    for (const tc of msg.tool_calls) {
      const args = typeof tc.function.arguments === "string"
        ? JSON.parse(tc.function.arguments)
        : tc.function.arguments
      inner.appendChild(createToolCard(tc.id, args.code))
    }
  }

  wrapper.appendChild(inner)
  return wrapper
}

function createToolResultMessage(msg) {
  // Tool results are rendered inline in the tool card, not as separate messages
  return null
}

export function createToolCard(toolCallId, code, result, isError, graphics) {
  const card = document.createElement("div")
  card.className = `tool-card${isError ? " collapsed" : ""}`
  card.dataset.toolCallId = toolCallId

  const header = document.createElement("div")
  header.className = "tool-card-header"
  header.innerHTML = `
    <svg class="chevron w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
    <span>Wolfram Language</span>
    ${result === undefined ? '<div class="spinner"></div>' : isError ? '<svg class="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>' : '<svg class="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>'}
  `
  header.addEventListener("click", () => card.classList.toggle("collapsed"))
  card.appendChild(header)

  const body = document.createElement("div")
  body.className = "tool-card-body"

  const codeEl = document.createElement("div")
  codeEl.className = "tool-card-code"
  codeEl.innerHTML = highlightCode(code, "mathematica")
  body.appendChild(codeEl)

  if (result !== undefined) {
    if (graphics) {
      const graphicsEl = document.createElement("div")
      graphicsEl.className = "tool-card-result graphics"
      graphicsEl.innerHTML = DOMPurify.sanitize(graphics)
      body.appendChild(graphicsEl)
    } else {
      const resultEl = document.createElement("div")
      resultEl.className = `tool-card-result${isError ? " error" : ""}`
      resultEl.textContent = result
      body.appendChild(resultEl)
    }
  }

  card.appendChild(body)
  return card
}

export function updateToolCard(toolCallId, result, isError, graphics, warnings) {
  const card = document.querySelector(`.tool-card[data-tool-call-id="${toolCallId}"]`)
  if (!card) return

  if (isError) card.classList.add("collapsed")

  // Replace spinner with check/error icon
  const header = card.querySelector(".tool-card-header")
  const spinner = header.querySelector(".spinner")
  if (spinner) {
    spinner.outerHTML = isError
      ? '<svg class="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>'
      : '<svg class="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>'
  }

  const body = card.querySelector(".tool-card-body")

  if (warnings) {
    const warningEl = document.createElement("div")
    warningEl.className = "tool-card-result warning"
    warningEl.textContent = warnings
    body.appendChild(warningEl)
  }

  if (graphics) {
    const graphicsEl = document.createElement("div")
    graphicsEl.className = "tool-card-result graphics"
    graphicsEl.innerHTML = DOMPurify.sanitize(graphics)
    body.appendChild(graphicsEl)
  } else {
    const resultEl = document.createElement("div")
    resultEl.className = `tool-card-result${isError ? " error" : ""}`
    resultEl.textContent = result
    body.appendChild(resultEl)
  }
}

export function createStreamingAssistant() {
  const wrapper = document.createElement("div")
  wrapper.className = "px-4 py-3"
  wrapper.id = "streaming-msg"

  const inner = document.createElement("div")
  inner.className = "max-w-3xl mx-auto"

  const contentEl = document.createElement("div")
  contentEl.className = "message-content text-sm streaming-cursor"
  inner.appendChild(contentEl)

  wrapper.appendChild(inner)
  return wrapper
}

export function appendStreamingText(text) {
  const el = document.querySelector("#streaming-msg .message-content")
  if (!el) return
  el._rawText = (el._rawText || "") + text
  el.innerHTML = renderMarkdown(el._rawText)
  el.classList.add("streaming-cursor")
}

export function finalizeStreamingMessage() {
  const el = document.querySelector("#streaming-msg .message-content")
  if (el) el.classList.remove("streaming-cursor")
  const wrapper = document.getElementById("streaming-msg")
  if (wrapper) wrapper.removeAttribute("id")
}

export function appendStreamingToolCard(toolCallId, code) {
  const wrapper = document.getElementById("streaming-msg")
  if (!wrapper) return
  const inner = wrapper.querySelector(".max-w-3xl")
  if (!inner) return

  // Remove cursor from text
  const contentEl = inner.querySelector(".message-content")
  if (contentEl) contentEl.classList.remove("streaming-cursor")

  inner.appendChild(createToolCard(toolCallId, code))
}

export function showToast(message, type = "error") {
  const container = document.getElementById("toast-container")
  const toast = document.createElement("div")
  const colors = {
    error: "bg-red-600 text-white",
    success: "bg-green-600 text-white",
    info: "bg-gray-800 text-white",
  }
  toast.className = `toast ${colors[type] || colors.info}`
  toast.textContent = message
  container.appendChild(toast)
  setTimeout(() => toast.remove(), 5000)
}

export function scrollToBottom() {
  const el = document.getElementById("messages")
  el.scrollTop = el.scrollHeight
}
