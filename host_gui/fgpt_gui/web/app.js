"use strict";

// fpGPT host GUI frontend.
// Talks to the stdlib host bridge over SSE (/api/events) for the streaming
// reply and POST (/api/prompt, /api/reset) for commands.

const el = (id) => document.getElementById(id);
const chat = el("chat");
const promptBox = el("prompt");
const sendBtn = el("send");
const resetBtn = el("reset");
const exportBtn = el("export");

const state = {
  config: null,
  params: null,
  charMin: 32,
  charMax: 126,
  genTokens: 16,
  busy: false,
  current: null, // { node, text }
  transcript: [], // { role, text }
};

function badge(id, text) {
  el(id).textContent = text;
}

function setConnected(online) {
  el("conn-dot").classList.toggle("online", online);
}

function setState(text, cls) {
  const node = el("state-text");
  node.textContent = text;
  node.className = "state " + (cls || "");
}

function addMessage(role, text, cls) {
  const node = document.createElement("div");
  node.className = "msg " + (cls || role);
  const label = document.createElement("span");
  label.className = "role";
  label.textContent = role;
  node.appendChild(label);
  const body = document.createElement("span");
  body.className = "body";
  body.textContent = text || "";
  node.appendChild(body);
  chat.appendChild(node);
  chat.scrollTop = chat.scrollHeight;
  return { node, body };
}

function appendToken(char) {
  if (!state.current) {
    state.current = addMessage("assistant", "");
  }
  state.current.body.textContent += char;
  state.current.node.classList.add("cursor");
  chat.scrollTop = chat.scrollHeight;
}

function endReply(reason) {
  if (state.current) {
    state.current.node.classList.remove("cursor");
    state.transcript.push({ role: "assistant", text: state.current.body.textContent });
    state.current = null;
  }
  if (reason && reason !== "complete" && reason !== "ok") {
    addMessage("system", "reply ended: " + reason, "system");
  }
}

function handleEvent(event) {
  switch (event.type) {
    case "hello":
      state.config = event.config;
      applyConfig(event.config);
      setConnected(true);
      setState("idle", "idle");
      break;
    case "info":
      if (event.params) {
        state.params = event.params;
        applyParams(event.params);
      }
      break;
    case "status":
      state.busy = !!event.busy;
      sendBtn.disabled = state.busy;
      if (event.state === "error") {
        setState("error", "error");
      } else if (state.busy) {
        setState("generating\u2026", "busy");
      } else {
        setState("idle", "idle");
      }
      break;
    case "token":
      appendToken(event.char);
      break;
    case "reply_end":
      endReply(event.reason);
      break;
    case "error":
      addMessage("system", "error: " + event.message, "system error");
      setState("error", "error");
      break;
    case "ready":
      addMessage("system", "board ready (protocol v" + event.protocol_version + ")", "system");
      break;
    default:
      break;
  }
}

function applyConfig(config) {
  badge("badge-transport", "transport: " + config.transport
    + (config.mock_style ? " (" + config.mock_style + ")" : ""));
  badge("badge-mode", "mode: " + config.mode);
  state.genTokens = config.gen_tokens;
  if (config.params) applyParams(config.params);
  if (config.warnings && config.warnings.length) {
    config.warnings.forEach((w) => addMessage("system", "warning: " + w, "system"));
  }
  el("gen-info").textContent = "reply length: " + config.gen_tokens + " chars";
}

function applyParams(params) {
  state.charMin = params.char_min;
  state.charMax = params.char_max;
  badge("badge-model", "model: d" + params.d_model + " L" + params.num_layers
    + " V" + params.vocab_size + " T" + params.max_seq_len);
  el("char-range").textContent =
    "supported: " + String.fromCharCode(params.char_min) + "\u2013"
    + String.fromCharCode(params.char_max);
}

function countInvalid() {
  const text = promptBox.value;
  let invalid = 0;
  for (const ch of text) {
    const code = ch.codePointAt(0);
    if (ch === "\n") continue;
    if (code < state.charMin || code > state.charMax) invalid++;
  }
  const total = text.replace(/\n/g, "").length;
  el("char-count").textContent = total + " chars"
    + (invalid ? " \u00b7 " + invalid + " unsupported" : "");
  el("char-count").style.color = invalid ? "var(--warn)" : "";
}

async function sendPrompt() {
  const text = promptBox.value;
  if (!text.trim() || state.busy) return;
  addMessage("user", text);
  state.transcript.push({ role: "user", text });
  promptBox.value = "";
  countInvalid();
  sendBtn.disabled = true;
  try {
    const response = await fetch("/api/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    if (!data.ok) {
      addMessage("system", "not sent: " + (data.error || "unknown error"), "system error");
      sendBtn.disabled = false;
    }
  } catch (err) {
    addMessage("system", "request failed: " + err, "system error");
    sendBtn.disabled = false;
  }
}

async function resetSession() {
  state.current = null;
  try {
    await fetch("/api/reset", { method: "POST" });
  } catch (err) {
    // ignore; the stream status will reflect reality
  }
}

function exportTranscript() {
  const lines = state.transcript.map(
    (m) => (m.role === "user" ? "> " : "< ") + m.text
  );
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "fpgpt-transcript.txt";
  a.click();
  URL.revokeObjectURL(url);
}

function subscribe() {
  const events = new EventSource("/api/events");
  events.onopen = () => setConnected(true);
  events.onerror = () => setConnected(false);
  events.onmessage = (message) => {
    try {
      handleEvent(JSON.parse(message.data));
    } catch (err) {
      console.error("bad event", message.data, err);
    }
  };
}

function init() {
  addMessage("system",
    "Connected to the host bridge. Type a prompt; the FPGA streams the reply.",
    "system");
  sendBtn.addEventListener("click", sendPrompt);
  resetBtn.addEventListener("click", resetSession);
  exportBtn.addEventListener("click", exportTranscript);
  promptBox.addEventListener("input", countInvalid);
  promptBox.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendPrompt();
    }
  });
  countInvalid();
  subscribe();
}

init();