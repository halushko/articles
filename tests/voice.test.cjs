const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

const script = fs.readFileSync(path.join(__dirname, '../web/voice.js'), 'utf8');

function setup({ recognitionSupported = true, speechSupported = true } = {}) {
  const timers = [];
  const spoken = [], form = { submits: 0, requestSubmit() { this.submits++; } };
  const classes = () => ({ toggle() {}, add() {}, remove() {} });
  const element = (value = '') => ({
    value, textContent: '', hidden: true, checked: true, disabled: false,
    classList: classes(), listeners: {}, attrs: {},
    addEventListener(event, callback) { this.listeners[event] = callback; },
    setAttribute(name, value) { this.attrs[name] = value; },
    click() { this.listeners.click?.(); },
    change() { this.listeners.change?.(); }
  });
  const input = element(''); input.maxLength = 3000;
  const button = element(), language = element('ru-RU'), status = element();
  const outputToggle = element(), outputStatus = element();
  const store = new Map();
  let recognizer;
  class FakeRecognition {
    constructor() { recognizer = this; this.starts = 0; this.stops = 0; this.aborts = 0; }
    start() { this.starts++; }
    stop() { this.stops++; this.onend?.(); }
    abort() { this.aborts++; this.onend?.(); }
    result(text, isFinal = true) {
      this.onresult({results: [{0: {transcript: text}, isFinal}]});
    }
    end() { this.onend?.(); }
  }
  class FakeUtterance { constructor(text) { this.text = text; } }
  const synthesis = {
    canceled: 0,
    cancel() { this.canceled++; },
    getVoices() { return [{lang:'ru-RU',name:'test-voice'}]; },
    speak(utterance) { spoken.push(utterance); }
  };
  const root = {
    ...(recognitionSupported ? {SpeechRecognition:FakeRecognition} : {}),
    ...(speechSupported ? {speechSynthesis:synthesis,SpeechSynthesisUtterance:FakeUtterance} : {}),
    localStorage: { getItem:key=>store.get(key), setItem:(key,value)=>store.set(key,value) },
    setTimeout(callback) { timers.push(callback); return timers.length; },
    clearTimeout(id) { timers[id-1] = null; }
  };
  vm.runInNewContext(script, {window:root});
  let busy = false;
  const voice = root.JarvisVoice.initVoice({
    input, button, language, status, form, outputToggle, outputStatus,
    isBusy:() => busy
  });
  return {
    input, button, language, status, form, voice, spoken, synthesis,
    outputToggle, outputStatus,
    recognition: () => recognizer,
    nextTimer: () => { while (timers.length) { const fn=timers.shift(); if (fn) return fn(); } },
    setBusy:v => {busy=v;}
  };
}

test('short pause does not submit and dictation survives browser end/restart', () => {
  const h = setup();
  h.button.click();
  assert.equal(h.recognition().starts, 1);
  assert.equal(h.recognition().continuous, true);
  h.recognition().result('Включи свет');
  h.recognition().end(); // browser ends session at a short pause
  assert.equal(h.form.submits, 0);
  assert.equal(h.input.value, 'Включи свет');
  assert.equal(h.button.attrs['aria-pressed'], 'true');
  h.nextTimer();
  assert.equal(h.recognition().starts, 2);
  h.recognition().result('в ванной');
  h.button.click(); // user, not browser silence, decides to send
  assert.equal(h.input.value, 'Включи свет в ванной');
  assert.equal(h.form.submits, 1);
  assert.equal(h.button.attrs['aria-pressed'], 'false');
});

test('no-speech is not a fatal condition; manual stop still submits', () => {
  const h = setup();
  h.button.click();
  h.recognition().onerror({error:'no-speech'});
  h.recognition().end();
  assert.equal(h.form.submits, 0);
  h.nextTimer();
  h.recognition().result('Заряд Мелкого');
  h.button.click();
  assert.equal(h.input.value, 'Заряд Мелкого');
  assert.equal(h.form.submits, 1);
});

test('manual send during dictation never submits twice or recreates old draft', () => {
  const h = setup();
  h.button.click();
  h.recognition().result('Я уже сказал',true);
  h.voice.stopForManualSend();
  h.input.value = '';
  assert.equal(h.form.submits, 0);
  assert.equal(h.input.value, '');
});

test('microphone starts by canceling speech; replies speak by default', () => {
  const h = setup();
  assert.equal(h.outputToggle.checked,true);
  assert.equal(h.voice.speak('Привет, Дима!'),true);
  assert.equal(h.spoken.length,1);
  assert.equal(h.spoken[0].text,'Привет, Дима!');
  assert.equal(h.spoken[0].lang,'ru-RU');
  h.button.click();
  assert.ok(h.synthesis.canceled >= 2);
  assert.equal(h.voice.speak('Пока слушаю'), false);
});

test('user can mute automatic speech but replay explicitly', () => {
  const h = setup();
  h.outputToggle.checked=false;
  h.outputToggle.change();
  assert.equal(h.voice.speak('Не читать автоматически'),false);
  assert.equal(h.spoken.length,0);
  assert.equal(h.voice.speak('Повтори',true),true);
  assert.equal(h.spoken[0].text,'Повтори');
});

test('cancelled multi-chunk speech never resumes stale utterances', () => {
  const h = setup();
  h.voice.speak('Длинный текст. '.repeat(55));
  const first=h.spoken[0];
  h.voice.cancelSpeech();
  first.onend();
  assert.equal(h.spoken.length,1);
});

test('browser without recognition still allows TTS and manual typing', () => {
  const h = setup({recognitionSupported:false});
  assert.equal(h.voice.isSupported,false);
  assert.equal(h.button.disabled,true);
  assert.equal(h.voice.speak('Я могу говорить'),true);
  assert.equal(h.spoken.length,1);
});

test('speech strips Markdown code fences, backticks and source markers but keeps explanation', () => {
  const h = setup();
  const original = [
    '## REST API', '',
    'Сделай запрос по `/api/states/<entity_id>/` [1].',
    'Пример команды:',
    '```bash',
    'curl -H "Authorization: Bearer TOKEN" http://homeassistant.local:8123/api/',
    '```',
    'Документация: [Home Assistant](https://www.home-assistant.io/integrations/api/).'
  ].join('\n');
  assert.equal(h.voice.speak(original), true);
  const spoken = h.spoken[0].text;
  assert.match(spoken, /рест эй пи ай/i);
  assert.match(spoken, /Пример кода показан на экране/);
  assert.match(spoken, /Home Assistant/);
  assert.doesNotMatch(spoken, /`|curl|Bearer|TOKEN|\/api\/states|\[1\]|https?:\/\//i);
  assert.doesNotMatch(spoken, /обратный апостроф/i);
});

test('plain text survives spoken rendering; inline link becomes readable label', () => {
  const h = setup();
  assert.equal(h.voice.speak('Привет! Подробности в [документации](https://example.org/docs).'),true);
  const result=h.spoken[0].text;
  assert.match(result,/Привет!/);
  assert.match(result,/документации/);
  assert.doesNotMatch(result,/https|\[|\]/);
});

test('Markdown table is referenced once instead of reading pipes and cells', () => {
  const h = setup();
  const original=['Сравнение:', '| Режим | Цена |','|---|---|',
                  '| Локально | 0 |','| Интернет | 0 |','Итог — бесплатно.'].join('\n');
  h.voice.speak(original);
  const rendered=h.spoken.map(x=>x.text).join(' ');
  assert.match(rendered,/Таблица с подробностями показана на экране/);
  assert.doesNotMatch(rendered,/\|/);
  assert.match(rendered,/бесплатно/);
});
