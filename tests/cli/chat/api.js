const TOOL_DEFINITION = {
  type: "function",
  function: {
    name: "evaluate_wolfram",
    description:
      "Evaluate Wolfram Language code using the Woxi interpreter. " +
      "State persists across calls within this conversation. " +
      "Use this to compute mathematical expressions, manipulate lists, " +
      "do symbolic algebra, solve equations, etc.",
    parameters: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "Wolfram Language code to evaluate",
        },
      },
      required: ["code"],
    },
  },
}

const ANTHROPIC_TOOL = {
  name: "evaluate_wolfram",
  description: TOOL_DEFINITION.function.description,
  input_schema: TOOL_DEFINITION.function.parameters,
}

/**
 * Unified interface for sending messages to LLM providers.
 *
 * @param {Array} messages - Messages in OpenAI format
 * @param {Object} opts
 * @param {string} opts.provider - "openai" or "anthropic"
 * @param {string} opts.apiKey
 * @param {(text: string) => void} opts.onToken - Incremental text callback
 * @param {(toolCall: {id: string, name: string, arguments: object}) => void} opts.onToolCall
 * @param {(message: {role: string, content?: string, tool_calls?: Array}) => void} opts.onDone
 * @param {(error: Error) => void} opts.onError
 * @param {AbortSignal} opts.signal
 */
export async function sendMessage(messages, opts) {
  if (opts.provider === "anthropic") {
    return sendAnthropic(messages, opts)
  }
  return sendOpenAI(messages, opts)
}

// --- OpenAI ---

async function sendOpenAI(messages, { apiKey, onToken, onToolCall, onDone, onError, signal }) {
  try {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "gpt-4o",
        messages,
        tools: [TOOL_DEFINITION],
        stream: true,
      }),
      signal,
    })

    if (!response.ok) {
      const body = await response.text()
      throw new Error(`OpenAI API error ${response.status}: ${body}`)
    }

    let textContent = ""
    let toolCalls = []
    let currentToolCall = null

    await parseSSE(response.body, (data) => {
      if (data === "[DONE]") return

      const chunk = JSON.parse(data)
      const delta = chunk.choices?.[0]?.delta
      const finishReason = chunk.choices?.[0]?.finish_reason

      if (!delta) return

      // Text content
      if (delta.content) {
        textContent += delta.content
        onToken(delta.content)
      }

      // Tool call deltas
      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          if (tc.id) {
            // New tool call
            currentToolCall = { id: tc.id, name: tc.function?.name || "", arguments: "" }
            toolCalls.push(currentToolCall)
          }
          if (tc.function?.name && currentToolCall) {
            currentToolCall.name = tc.function.name
          }
          if (tc.function?.arguments && currentToolCall) {
            currentToolCall.arguments += tc.function.arguments
          }
        }
      }

      if (finishReason === "tool_calls") {
        for (const tc of toolCalls) {
          try {
            const args = JSON.parse(tc.arguments)
            onToolCall({ id: tc.id, name: tc.name, arguments: args })
          } catch (e) {
            onError(new Error(`Failed to parse tool arguments: ${tc.arguments}`))
          }
        }
      }
    })

    // Build the complete assistant message
    const msg = { role: "assistant", content: textContent || null }
    if (toolCalls.length > 0) {
      msg.tool_calls = toolCalls.map((tc) => ({
        id: tc.id,
        type: "function",
        function: { name: tc.name, arguments: tc.arguments },
      }))
    }
    onDone(msg)
  } catch (e) {
    if (e.name === "AbortError") return
    onError(e)
  }
}

// --- Anthropic ---

function toAnthropicMessages(messages) {
  const result = []
  for (const msg of messages) {
    if (msg.role === "system") continue

    if (msg.role === "user") {
      result.push({ role: "user", content: msg.content })
      continue
    }

    if (msg.role === "assistant") {
      const content = []
      if (msg.content) {
        content.push({ type: "text", text: msg.content })
      }
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          const input = typeof tc.function.arguments === "string"
            ? JSON.parse(tc.function.arguments)
            : tc.function.arguments
          content.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input,
          })
        }
      }
      result.push({ role: "assistant", content })
      continue
    }

    if (msg.role === "tool") {
      // Check if last message is user with tool_result content
      const last = result[result.length - 1]
      const block = {
        type: "tool_result",
        tool_use_id: msg.tool_call_id,
        content: msg.content,
      }
      if (last && last.role === "user" && Array.isArray(last.content)) {
        last.content.push(block)
      } else {
        result.push({ role: "user", content: [block] })
      }
    }
  }
  return result
}

async function sendAnthropic(messages, { apiKey, onToken, onToolCall, onDone, onError, signal }) {
  const systemMsg = messages.find((m) => m.role === "system")
  const anthropicMessages = toAnthropicMessages(messages)

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true",
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 8192,
        system: systemMsg?.content || "",
        messages: anthropicMessages,
        tools: [ANTHROPIC_TOOL],
        stream: true,
      }),
      signal,
    })

    if (!response.ok) {
      const body = await response.text()
      throw new Error(`Anthropic API error ${response.status}: ${body}`)
    }

    let textContent = ""
    let toolCalls = []
    let currentToolUse = null
    let currentToolInput = ""

    await parseSSE(response.body, (data) => {
      const event = JSON.parse(data)

      if (event.type === "content_block_start") {
        const block = event.content_block
        if (block.type === "tool_use") {
          currentToolUse = { id: block.id, name: block.name }
          currentToolInput = ""
        }
      }

      if (event.type === "content_block_delta") {
        const delta = event.delta
        if (delta.type === "text_delta") {
          textContent += delta.text
          onToken(delta.text)
        }
        if (delta.type === "input_json_delta") {
          currentToolInput += delta.partial_json
        }
      }

      if (event.type === "content_block_stop") {
        if (currentToolUse) {
          try {
            const args = currentToolInput ? JSON.parse(currentToolInput) : {}
            const tc = { id: currentToolUse.id, name: currentToolUse.name, arguments: args }
            toolCalls.push(tc)
            onToolCall(tc)
          } catch (e) {
            onError(new Error(`Failed to parse tool input: ${currentToolInput}`))
          }
          currentToolUse = null
          currentToolInput = ""
        }
      }
    })

    // Build the complete assistant message in OpenAI format for storage
    const msg = { role: "assistant", content: textContent || null }
    if (toolCalls.length > 0) {
      msg.tool_calls = toolCalls.map((tc) => ({
        id: tc.id,
        type: "function",
        function: { name: tc.name, arguments: JSON.stringify(tc.arguments) },
      }))
    }
    onDone(msg)
  } catch (e) {
    if (e.name === "AbortError") return
    onError(e)
  }
}

// --- SSE Parser ---

async function parseSSE(body, onData) {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split("\n")
    buffer = lines.pop()

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith(":")) continue
      if (trimmed.startsWith("data: ")) {
        const data = trimmed.slice(6)
        if (data === "[DONE]") return
        onData(data)
      }
    }
  }

  // Process remaining buffer
  if (buffer.trim().startsWith("data: ")) {
    const data = buffer.trim().slice(6)
    if (data !== "[DONE]") onData(data)
  }
}
