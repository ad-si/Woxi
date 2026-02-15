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
  if (msg.role === "assistant") return createAssistantMessage(msg, index)
  if (msg.role === "tool") return createToolResultMessage(msg)
  return null
}

function createUserMessage(msg, index) {
  const el = document.createElement("div")
  el.className = "user-message-row flex flex-col items-end px-4 py-3"
  if (index !== undefined) el.dataset.msgIndex = index

  const bubble = document.createElement("div")
  bubble.className = "max-w-[80%] px-4 py-2.5 rounded-2xl bg-blue-600 text-white text-sm whitespace-pre-wrap"

  // Render attachments inside the bubble
  if (msg.attachments && msg.attachments.length > 0) {
    // Image thumbnails first
    for (const att of msg.attachments) {
      if (att.type === "image") {
        const img = document.createElement("img")
        img.className = "msg-attachment-thumb"
        img.src = `data:${att.mediaType};base64,${att.data}`
        img.alt = att.name
        bubble.appendChild(img)
      }
    }

    // Text file labels
    const textAtts = msg.attachments.filter((a) => a.type === "text")
    if (textAtts.length > 0) {
      const labels = document.createElement("div")
      labels.className = "msg-attachments"
      for (const att of textAtts) {
        const label = document.createElement("span")
        label.className = "msg-attachment-label"
        label.innerHTML = `<svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>`
        const nameSpan = document.createElement("span")
        nameSpan.textContent = att.name
        label.appendChild(nameSpan)
        labels.appendChild(label)
      }
      bubble.appendChild(labels)
    }
  }

  const textNode = document.createTextNode(msg.content)
  bubble.appendChild(textNode)

  el.appendChild(bubble)
  el.innerHTML += `
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

function createAssistantMessage(msg, index) {
  const wrapper = document.createElement("div")
  const isTextOnly = msg.content && !msg.tool_calls
  wrapper.className = isTextOnly ? "assistant-message-row px-4 py-3" : "px-4 py-3"
  if (isTextOnly && index !== undefined) wrapper.dataset.msgIndex = index

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

  // Action buttons for text-only assistant messages (final responses)
  if (isTextOnly) {
    const actions = document.createElement("div")
    actions.className = "assistant-msg-actions"
    actions.innerHTML = `
      <button class="msg-action-btn" data-action="copy-assistant" title="Copy">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" stroke-width="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" stroke-width="2"/></svg>
      </button>
      <button class="msg-action-btn" data-action="retry-assistant" title="Try again">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M1 4v6h6"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>
      </button>
    `
    inner.appendChild(actions)
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
    <span class="flex-1"></span>
    ${result !== undefined ? `<button class="tool-header-btn copy-result-btn" title="Copy result"><svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" stroke-width="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" stroke-width="2"/></svg></button><button class="tool-header-btn recalc-btn" title="Recalculate"><svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M1 4v6h6"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg></button>` : ''}
  `
  header.addEventListener("click", (e) => {
    if (e.target.closest(".tool-header-btn")) return
    card.classList.toggle("collapsed")
  })
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

  // Replace spinner with check/error icon + add recalculate button
  const header = card.querySelector(".tool-card-header")
  const spinner = header.querySelector(".spinner")
  if (spinner) {
    spinner.outerHTML = isError
      ? '<svg class="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>'
      : '<svg class="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>'

    // Add spacer + copy + recalculate buttons
    if (!header.querySelector(".recalc-btn")) {
      const spacer = document.createElement("span")
      spacer.className = "flex-1"
      header.appendChild(spacer)
      const copyBtn = document.createElement("button")
      copyBtn.className = "tool-header-btn copy-result-btn"
      copyBtn.title = "Copy result"
      copyBtn.innerHTML = '<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" stroke-width="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" stroke-width="2"/></svg>'
      header.appendChild(copyBtn)
      const recalcBtn = document.createElement("button")
      recalcBtn.className = "tool-header-btn recalc-btn"
      recalcBtn.title = "Recalculate"
      recalcBtn.innerHTML = '<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M1 4v6h6"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>'
      header.appendChild(recalcBtn)
    }
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

export function finalizeStreamingMessage(msgIndex) {
  const el = document.querySelector("#streaming-msg .message-content")
  if (el) {
    el.classList.remove("streaming-cursor")
    // Remove empty content div (e.g. when assistant only made tool calls)
    if (!el._rawText) el.remove()
  }
  const wrapper = document.getElementById("streaming-msg")
  if (!wrapper) return
  wrapper.removeAttribute("id")

  // Text-only messages (no tool cards) get copy/retry actions
  const hasToolCards = wrapper.querySelector(".tool-card")
  const hasText = el && el._rawText
  if (msgIndex !== undefined && hasText && !hasToolCards) {
    wrapper.classList.add("assistant-message-row")
    wrapper.dataset.msgIndex = msgIndex
    const inner = wrapper.querySelector(".max-w-3xl")
    if (inner && !inner.querySelector(".assistant-msg-actions")) {
      const actions = document.createElement("div")
      actions.className = "assistant-msg-actions"
      actions.innerHTML = `
        <button class="msg-action-btn" data-action="copy-assistant" title="Copy">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" stroke-width="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" stroke-width="2"/></svg>
        </button>
        <button class="msg-action-btn" data-action="retry-assistant" title="Try again">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M1 4v6h6"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>
        </button>
      `
      inner.appendChild(actions)
    }
  }
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
