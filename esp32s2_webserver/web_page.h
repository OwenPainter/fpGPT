// fpGPT ESP32-S2 webserver — embedded single-page chat UI.
//
// Served from flash with send_P(), so no SPIFFS/LittleFS filesystem is needed.

#pragma once

#include <Arduino.h>

static const char INDEX_HTML[] PROGMEM = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fpGPT — ESP32-S2</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 15px/1.5 system-ui, sans-serif; background: #10141a; color: #d7dde5; }
  header { padding: 14px 18px; border-bottom: 1px solid #232a34; display: flex; gap: 12px; align-items: baseline; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  #status { font-size: 12px; color: #7d8a99; margin-left: auto; }
  main { max-width: 760px; margin: 0 auto; padding: 18px; display: flex; flex-direction: column; gap: 12px; }
  #out { min-height: 320px; max-height: 60vh; overflow: auto; white-space: pre-wrap;
         background: #0b0e13; border: 1px solid #232a34; border-radius: 10px; padding: 14px; }
  .prompt { color: #6fb3ff; }
  .err { color: #ff7b72; }
  form { display: flex; gap: 10px; }
  textarea { flex: 1; resize: vertical; min-height: 52px; background: #0b0e13; color: inherit;
             border: 1px solid #232a34; border-radius: 10px; padding: 10px; font: inherit; }
  button { background: #2f6fed; color: #fff; border: 0; border-radius: 10px; padding: 0 18px;
           font: inherit; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
</style>
</head>
<body>
<header>
  <h1>fpGPT</h1>
  <span id="status">connecting…</span>
</header>
<main>
  <div id="out"></div>
  <form id="form">
    <textarea id="prompt" placeholder="Type a prompt and press Enter…" autofocus></textarea>
    <button id="send" type="submit">Send</button>
  </form>
</main>
<script>
const out = document.getElementById('out');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');

function append(text, cls) {
  const span = document.createElement('span');
  if (cls) span.className = cls;
  span.textContent = text;
  out.appendChild(span);
  out.scrollTop = out.scrollHeight;
}

async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const j = await r.json();
    statusEl.textContent = (j.wifi ? 'wifi ' + j.ip + ' (' + j.rssi + ' dBm)' : 'wifi down')
      + ' · uart ' + j.baud + ' · ' + (j.framed ? 'framed' : 'raw')
      + (j.mock ? ' · mock' : '') + ' · err ' + j.decoder_errors;
  } catch (e) {
    statusEl.textContent = 'server unreachable';
  }
}

async function generate(prompt) {
  sendBtn.disabled = true;
  append('\n> ' + prompt + '\n', 'prompt');
  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      body: prompt,
    });
    if (!res.ok) { append('[http ' + res.status + ']\n', 'err'); return; }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      append(dec.decode(value, { stream: true }));
    }
  } catch (e) {
    append('\n[error: ' + e + ']\n', 'err');
  } finally {
    sendBtn.disabled = false;
    promptEl.focus();
  }
}

document.getElementById('form').addEventListener('submit', (ev) => {
  ev.preventDefault();
  const prompt = promptEl.value.trim();
  if (!prompt) return;
  promptEl.value = '';
  generate(prompt);
});

// Enter sends; Shift+Enter inserts a newline.
promptEl.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault();
    document.getElementById('form').requestSubmit();
  }
});

refreshStatus();
setInterval(refreshStatus, 3000);
</script>
</body>
</html>
)HTML";
