/* JARVIS voice: browser TTS + dictation that stays open across natural pauses.
   No audio is uploaded to JARVIS; Web Speech engines can use browser vendor services. */
(function (root) {
  'use strict';

  function prepareSpeechText(input) {
    /* Speech is a spoken summary, not a second rendering of Markdown.
       Original content, links and code remain untouched in the chat/export. */
    let text = String(input || '').replace(/\r\n?/g, '\n');
    text = text.replace(/^\s*(?:```|~~~)[^\n]*\n[\s\S]*?^\s*(?:```|~~~)\s*$/gm,
                        '\nПример кода показан на экране.\n');
    text = text.replace(/!\[([^\]]*)\]\(\s*[^)]+\)/g, '$1');
    text = text.replace(/\[([^\]]+)\]\(\s*(?:https?:\/\/|mailto:)[^)]+\)/gi, '$1');
    text = text.replace(/\bhttps?:\/\/[^\s<>)]*/gi, 'ссылка');
    // A source number is visual provenance, not part of the sentence.
    text = text.replace(/\s*\[\d{1,3}\]/g, '');
    text = text.replace(/^\s{0,3}#{1,6}\s*/gm, '');
    text = text.replace(/^\s{0,3}>\s?/gm, '');
    text = text.replace(/^\s*[-*+]\s+/gm, '');
    text = text.replace(/^\s*\d+[.)]\s+/gm, '');
    // A Markdown table is useful visually, but should not be read cell by cell.
    text = text.replace(/(?:^\s*\|[^\n]*\|\s*$\n?){2,}/gm,
                        '\nТаблица с подробностями показана на экране.\n');
    text = text.replace(/[`]{1,2}([^`\n]+)[`]{1,2}/g,(_,value)=>{
      const token=value.trim();
      if (/^(?:\/?api\/|\/|[\w.-]+\/[\w./<>-]+|[\w.-]+\.[\w./<>-]+$)/i.test(token)) {
        return 'путь в API';
      }
      if (/^[\w.-]{1,50}$/.test(token)) return token.replace(/_/g,' ');
      return 'фрагмент кода';
    });
    text = text.replace(/<(?:[\w-]{2,50})>/g,'идентификатор');
    text = text.replace(/\*\*|__|~~/g, '');
    text = text.replace(/\*([^*\n]+)\*/g,'$1');
    text = text.replace(/(^|\s)_([^_\n]+)_(?=\s|$)/g,'$1$2');
    text = text.replace(/(?:^\s*[-|:]\s*[-|:]\s*$)/gm,'');
    text = text.replace(/\bREST\s+API\b/gi,'рест эй пи ай');
    text = text.replace(/\bAPI\b/gi,'эй пи ай');
    text = text.replace(/\bHTTP\b/gi,'эйч ти ти пи');
    text = text.replace(/\bHTTPS\b/gi,'эйч ти ти пи эс');
    text = text.replace(/\bURL\b/gi,'ю эр эл');
    text = text.replace(/\bJSON\b/gi,'джейсон');
    text = text.replace(/\bSQL\b/gi,'эс кью эл');
    text = text.replace(/[<>={}\\^|#]/g,' ');
    text = text.replace(/[→⇒]/g, ' приводит к ');
    text = text.replace(/[\t ]+/g,' ').replace(/\n\s*\n+/g,'. ');
    text = text.replace(/\s*\n\s*/g,'. ');
    return text.replace(/\s+([,.!?;:])/g,'$1')
      .replace(/\.{2,}/g,'.').replace(/\s+/g,' ').trim();
  }

  function initVoice(opts) {
    const { input, button, language, status, form, outputToggle, outputStatus } = opts;
    const Recognition = root.SpeechRecognition || root.webkitSpeechRecognition;
    const synth = root.speechSynthesis || null;
    const Utterance = root.SpeechSynthesisUtterance || null;
    const canSpeak = !!(synth && Utterance);
    const canListen = !!Recognition;
    const key = 'jarvis.voice.output.v1';
    let enabled = true;
    try { enabled = root.localStorage.getItem(key) !== 'off'; } catch (_) {}
    outputToggle.checked = enabled;
    if (!canSpeak) {
      outputToggle.checked = false;
      outputToggle.disabled = true;
      outputStatus.textContent = 'Озвучивание недоступно в этом браузере.';
    }

    let recording = false, running = false, stopping = false;
    let draft = '', committed = '', segmentFinal = '', segmentInterim = '';
    let fatal = false, restartHandle = null, suppressSubmit = false;
    let audioUtterances = [];
    let speechEpoch = 0;

    function notice(value, error) {
      status.textContent = value || '';
      status.hidden = !value;
      status.classList.toggle('error', !!error);
    }
    function join(...parts) {
      return parts.map(s => String(s || '').trim()).filter(Boolean).join(' ').slice(0, input.maxLength);
    }
    function renderText(includeInterim) {
      input.value = join(draft, committed, segmentFinal, includeInterim ? segmentInterim : '');
    }
    function updateMic() {
      button.textContent = recording ? '⏹' : '🎤';
      button.setAttribute('aria-pressed', String(recording));
      button.setAttribute('aria-label', recording ?
        'Закончить диктовку и отправить' : 'Начать диктовку');
      button.title = recording ? 'Закончить диктовку и отправить' : 'Продиктовать команду';
      button.classList.toggle('is-listening', recording);
      language.disabled = recording;
    }
    function cancelSpeech() {
      speechEpoch++;
      if (canSpeak) {
        try { synth.cancel(); } catch (_) {}
      }
      audioUtterances = [];
      if (canSpeak) outputStatus.textContent = '';
    }
    function splitForSpeech(value) {
      const segments = [];
      // Short utterances avoid engines stopping on long, unbroken speech.
      let remaining = prepareSpeechText(value);
      while (remaining) {
        if (remaining.length <= 180) { segments.push(remaining); break; }
        const prefix = remaining.slice(0, 180);
        let cut = Math.max(prefix.lastIndexOf('. '), prefix.lastIndexOf('! '), prefix.lastIndexOf('? '));
        if (cut > 70) cut += 1;
        else {
          cut = prefix.lastIndexOf(' ');
          if (cut < 60) cut = 180;
        }
        segments.push(remaining.slice(0, cut).trim());
        remaining = remaining.slice(cut).trim();
      }
      return segments;
    }
    function speak(value, force = false) {
      if (!canSpeak || (!outputToggle.checked && !force) || recording || !value) return false;
      cancelSpeech();
      const parts = splitForSpeech(value);
      if (!parts.length) return false;
      const thisSpeech = speechEpoch;
      const lang = language.value === 'uk-UA' ? 'uk-UA' : 'ru-RU';
      const available = typeof synth.getVoices === 'function' ? synth.getVoices() : [];
      const voice = available.find(v => v.lang.toLowerCase() === lang.toLowerCase()) ||
                    available.find(v => v.lang.toLowerCase().startsWith(lang.slice(0,2).toLowerCase()));
      // Queue one utterance at a time; preserve objects for Chromium GC.
      let index = 0;
      function next() {
        if (thisSpeech !== speechEpoch) return;
        if (index >= parts.length) {
          outputStatus.textContent = '';
          audioUtterances = [];
          return;
        }
        const utterance = new Utterance(parts[index++]);
        utterance.lang = lang;
        utterance.rate = 1;
        if (voice) utterance.voice = voice;
        utterance.onend = next;
        utterance.onerror = () => {
          if (thisSpeech !== speechEpoch) return;
          outputStatus.textContent = 'Озвучивание прервано браузером.';
          audioUtterances = [];
        };
        audioUtterances = [utterance];
        try { synth.speak(utterance); }
        catch (_) {
          audioUtterances = [];
          outputStatus.textContent = 'Не удалось воспроизвести голос.';
        }
      }
      outputStatus.textContent = '🔊 Джарвис отвечает…';
      next();
      return true;
    }

    outputToggle.addEventListener('change', () => {
      try { root.localStorage.setItem(key, outputToggle.checked ? 'on' : 'off'); } catch (_) {}
      if (!outputToggle.checked) cancelSpeech();
    });

    if (!canListen) {
      button.disabled = true;
      button.title = 'Распознавание речи не поддерживается';
      notice('Распознавание речи недоступно в этом браузере (попробуй Chrome/Edge).', true);
      return { stopForManualSend: () => {}, cancelSpeech, speak, isSupported: false };
    }
    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    function clearRestart() {
      if (restartHandle !== null) {
        root.clearTimeout(restartHandle);
        restartHandle = null;
      }
    }
    function finish(shouldSend) {
      clearRestart();
      recording = false; running = false; stopping = false;
      updateMic();
      renderText(false);
      if (shouldSend && committed.trim()) {
        if (opts.isBusy()) {
          notice('Текст распознан, но Джарвис ещё отвечает. Нажми «Отправить».');
        } else {
          notice('Запись завершена. Отправляю…');
          form.requestSubmit();
        }
      } else if (shouldSend) {
        notice('Речь не распознана. Попробуй снова.', true);
      } else {
        notice('Диктовка остановлена, текст остался в поле.');
      }
    }
    function startEngine() {
      if (!recording || stopping || fatal || running) return;
      recognition.lang = language.value;
      try {
        recognition.start();
        running = true;
        notice(committed ? 'Продолжаю слушать… Закончи кнопкой ⏹.' :
          'Слушаю… Можно делать паузы. Закончи кнопкой ⏹.');
      } catch (err) {
        fatal = true;
        finish(false);
        notice('Браузер не разрешил продолжить запись. Текст сохранён; нажми «Отправить».', true);
      }
    }
    function commitSegment(includeInterim) {
      committed = join(committed, segmentFinal, includeInterim ? segmentInterim : '');
      segmentFinal = ''; segmentInterim = '';
      renderText(false);
    }
    recognition.onresult = event => {
      const finals = [], interim = [];
      for (let i = 0; i < event.results.length; i++) {
        const item = event.results[i];
        const text = String(item[0]?.transcript || '').trim();
        if (!text) continue;
        if (item.isFinal) finals.push(text);
        else interim.push(text);
      }
      segmentFinal = finals.join(' ').trim();
      segmentInterim = interim.join(' ').trim();
      renderText(true);
    };
    recognition.onerror = event => {
      // A browser service may end a recognition segment when the user pauses.
      // This is NOT a reason to send the message or lose the session.
      if (event.error === 'no-speech') {
        notice('Пауза — продолжаю слушать. Закончи кнопкой ⏹.');
        return;
      }
      if (event.error === 'aborted' && stopping) return;
      fatal = true;
      const explanations = {
        'not-allowed':'Нет разрешения на микрофон. Разреши его в настройках сайта.',
        'service-not-allowed':'Браузер запретил службу распознавания речи.',
        'audio-capture':'Микрофон не найден или занят.',
        'network':'Ошибка соединения со службой распознавания.',
        'language-not-supported':'Служба не поддерживает выбранный язык.'
      };
      notice(explanations[event.error] || 'Ошибка распознавания речи.', true);
    };
    recognition.onend = () => {
      running = false;
      if (stopping && suppressSubmit) {
        recording = false; stopping = false;
        clearRestart(); updateMic();
        return; // A manual send must not resurrect the old draft.
      }
      commitSegment(stopping);
      if (!recording) return;
      if (stopping) {
        finish(!suppressSubmit && !fatal);
        return;
      }
      if (fatal) {
        finish(false);
        notice('Распознавание остановлено из-за ошибки. Текст остался в поле.', true);
        return;
      }
      // Chrome/Edge sometimes auto-end even with continuous=true. Restart
      // the same dictation while the button is active; NEVER auto-submit.
      restartHandle = root.setTimeout(() => {
        restartHandle = null;
        startEngine();
      }, 350);
    };
    button.addEventListener('click', () => {
      if (opts.isBusy()) return;
      if (!recording) {
        cancelSpeech();
        draft = input.value.trim();
        committed = ''; segmentFinal = ''; segmentInterim = '';
        stopping = false; suppressSubmit = false; fatal = false; recording = true;
        updateMic();
        startEngine();
        return;
      }
      // Explicit Stop = end of the whole sentence, not an arbitrary pause.
      stopping = true;
      clearRestart();
      notice('Завершаю запись…');
      if (running) {
        try { recognition.stop(); } catch (_) { finish(true); }
      } else {
        commitSegment(true);
        finish(true);
      }
    });
    function stopForManualSend() {
      if (!recording) return;
      suppressSubmit = true; stopping = true;
      clearRestart();
      if (running) {
        try { recognition.abort(); } catch (_) { finish(false); }
      } else {
        finish(false);
      }
    }
    updateMic();
    return { stopForManualSend, cancelSpeech, speak, isSupported: true };
  }

  root.JarvisVoice = { initVoice, prepareSpeechText };
})(window);
