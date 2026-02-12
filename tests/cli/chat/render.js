import { Marked } from "https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js"
import DOMPurify from "https://cdn.jsdelivr.net/npm/dompurify/dist/purify.es.mjs"

const marked = new Marked({
  breaks: true,
  gfm: true,
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

export function createMessageElement(msg) {
  if (msg.role === "user") return createUserMessage(msg)
  if (msg.role === "assistant") return createAssistantMessage(msg)
  if (msg.role === "tool") return createToolResultMessage(msg)
  return null
}

function createUserMessage(msg) {
  const el = document.createElement("div")
  el.className = "flex justify-end px-4 py-3"
  el.innerHTML = `
    <div class="max-w-[80%] px-4 py-2.5 rounded-2xl bg-blue-600 text-white text-sm whitespace-pre-wrap">${escapeHtml(msg.content)}</div>
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
  card.className = "tool-card"
  card.dataset.toolCallId = toolCallId

  const header = document.createElement("div")
  header.className = "tool-card-header"
  header.innerHTML = `
    <svg class="chevron w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
    <span>Wolfram Language</span>
    ${result === undefined ? '<div class="spinner"></div>' : '<svg class="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>'}
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
    const resultEl = document.createElement("div")
    resultEl.className = `tool-card-result${isError ? " error" : ""}`
    resultEl.textContent = result
    body.appendChild(resultEl)

    if (graphics) {
      const graphicsEl = document.createElement("div")
      graphicsEl.className = "tool-card-result graphics"
      graphicsEl.innerHTML = DOMPurify.sanitize(graphics)
      body.appendChild(graphicsEl)
    }
  }

  card.appendChild(body)
  return card
}

export function updateToolCard(toolCallId, result, isError, graphics) {
  const card = document.querySelector(`.tool-card[data-tool-call-id="${toolCallId}"]`)
  if (!card) return

  // Replace spinner with check
  const header = card.querySelector(".tool-card-header")
  const spinner = header.querySelector(".spinner")
  if (spinner) {
    spinner.outerHTML = isError
      ? '<svg class="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>'
      : '<svg class="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>'
  }

  const body = card.querySelector(".tool-card-body")

  const resultEl = document.createElement("div")
  resultEl.className = `tool-card-result${isError ? " error" : ""}`
  resultEl.textContent = result
  body.appendChild(resultEl)

  if (graphics) {
    const graphicsEl = document.createElement("div")
    graphicsEl.className = "tool-card-result graphics"
    graphicsEl.innerHTML = DOMPurify.sanitize(graphics)
    body.appendChild(graphicsEl)
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
