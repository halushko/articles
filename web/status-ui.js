/* Observable pipeline state: never invent token-level thoughts or ETA.
 * The gateway publishes actual stages, the UI shows elapsed wait and silence.
 */
'use strict';
window.JarvisProgress = (() => {
  const root = document.getElementById('chat-progress');
  const step = document.getElementById('chat-progress-step');
  const elapsed = document.getElementById('chat-progress-elapsed');
  const hint = document.getElementById('chat-progress-hint');
  let interval = null, generation = 0, startAt = 0, finished = true;
  const labels = {
    received:'Запрос принят', routing:'Выбор маршрута',
    memory:'Проверка памяти SQLite', planning:'Qwen обрабатывает запрос',
    web_fetch:'Поиск и чтение страниц', web_answer:'Qwen изучает источники',
    saving:'Запись журнала', done:'Ответ готов',error:'Ошибка запроса',
    unknown:'Ожидание состояния сервера',expired:'Нет сведений о запросе'
  };
  const seconds = value => Math.max(0,Math.round(value || 0));
  function show(phase,detail,elapsedSeconds=0,phaseSeconds=0,finishedRequest=false) {
    root.hidden = false;
    root.classList.toggle('finished',finishedRequest);
    root.classList.toggle('failed',phase==='error');
    root.dataset.phase = phase;
    step.textContent = '● ' + (labels[phase] || 'Обработка') +
      (detail ? ' · ' + detail : '');
    elapsed.textContent = `⏱ ${seconds(elapsedSeconds)} с`;
    hint.textContent = finishedRequest ? '' :
      phaseSeconds >= 40 ? 'Этап длится дольше обычного. Сервер всё ещё может обрабатывать запрос; процент готовности неизвестен.' :
      'Показан текущий этап сервера, а не внутренние мысли модели.';
  }
  function stop() {
    if (interval !== null) clearInterval(interval);
    interval = null; generation++;
  }
  function begin(id,api) {
    stop();finished=false;startAt=Date.now();
    show('received','Запрос готовится',0);
    const mine=generation;
    let fetching=false,phase='received',detail='Запрос готовится',phaseAt=Date.now();
    async function tick(){
      if (mine!==generation||finished) return;
      if (!fetching) {
        fetching=true;
        try {
          const result=await api('/api/chat/status/'+encodeURIComponent(id));
          if (mine!==generation||finished) return;
          if (result.phase!=='unknown' && result.phase!=='expired') {
            if (phase!==result.phase) phaseAt=Date.now();
            phase=result.phase;detail=result.detail || '';
            if (result.running===false&&phase!=='done'&&phase!=='error') {
              phase='unknown';detail='Потеряно соединение с процессом запроса';
            }
          }
          show(phase,detail,(Date.now()-startAt)/1000,
               result.phase==='unknown' ? (Date.now()-phaseAt)/1000 :
               result.phase_elapsed_seconds || (Date.now()-phaseAt)/1000);
        } catch (_) {
          if (mine===generation&&!finished)
            show(phase,detail+' · связь со статусом временно недоступна',
                 (Date.now()-startAt)/1000,(Date.now()-phaseAt)/1000);
        } finally {fetching=false;}
      } else {
        show(phase,detail,(Date.now()-startAt)/1000,
             (Date.now()-phaseAt)/1000);
      }
    }
    interval=setInterval(tick,1100);
    tick();
  }
  function end(ok=true) {
    if (finished) return;
    finished=true;
    const elapsedSeconds=(Date.now()-startAt)/1000;
    stop();
    show(ok?'done':'error',ok?'Ответ получен':'См. сообщение об ошибке',
         elapsedSeconds,0,true);
    setTimeout(()=>{if(finished) root.hidden=true;},7000);
  }
  return {begin,end};
})();
