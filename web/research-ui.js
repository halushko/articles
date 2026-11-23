/* Research session UI. No model or API credentials are stored in localStorage:
 * only the UUID of the currently opened SQLite-backed conversation.
 */
'use strict';
window.JarvisResearchUI = (() => {
  const $ = id => document.getElementById(id);
  const storageKey = 'jarvis.activeResearchSession.v1';
  const webKey = 'jarvis.webPolicy.v1';
  let client = null, current = '', boot = null, lastQuestion = '';
  const own = () => Boolean(current);
  const alertStatus = text => { $('session-status').textContent = text; };

  async function ensure() {
    if (current) return current;
    if (boot) await boot;
    if (current) return current;
    await bootstrap();
    if (!current) throw new Error('Не удалось создать исследовательский диалог');
    return current;
  }

  async function list() {
    const result = await client.api('/api/sessions');
    const select = $('session-select');
    select.replaceChildren();
    for (const item of result.sessions || []) {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = `${item.title} · ${item.turns} реплик · ${item.updated_at} UTC`;
      select.append(option);
    }
    if (current) select.value = current;
    return result.sessions || [];
  }

  async function bootstrap() {
    const sessions = await list();
    const remembered = localStorage.getItem(storageKey);
    let chosen = sessions.find(x => x.id === remembered)?.id ||
                 sessions[0]?.id || '';
    if (!chosen) {
      const created = await client.api('/api/sessions', {
        method: 'POST', body: JSON.stringify({title: 'Новый диалог'})
      });
      chosen = created.id;
      await list();
    }
    await load(chosen);
  }

  function addSources(card, sources) {
    if (!sources?.length) return;
    const container = document.createElement('details');
    container.className = 'source-panel';
    const heading = document.createElement('summary');
    heading.textContent = `Источники и происхождение · ${sources.length}`;
    container.append(heading);
    for (const source of sources) {
      const row = document.createElement('div');
      row.className = 'source-entry';
      let link;
      try {
        const u = new URL(source.url);
        if (u.protocol === 'https:' && !u.username && !u.password) {
          link = document.createElement('a');
          link.href = u.href; link.target = '_blank'; link.rel = 'noopener noreferrer';
          link.textContent = source.title || u.hostname;
        }
      } catch (_) { /* manually pasted answers have no source URL */ }
      const title = document.createElement('strong');
      title.textContent = source.title || 'Недоверенный текст';
      if (link) row.append(link); else row.append(title);
      const state = document.createElement('small');
      state.textContent = source.status || 'результат поиска';
      const excerpt = document.createElement('p');
      excerpt.textContent = (source.snippet || source.page_text || '').slice(0, 450);
      row.append(state, excerpt); container.append(row);
    }
    card.append(container);
  }

  function addPermission(card, permission, originTurnId) {
    if (!permission?.query) return;
    // Old exports contain a query but no durable approval ID. They cannot
    // authorize new tool calls. Ask for a fresh request instead.
    if (!permission.id) {
      const warning=document.createElement('p');
      warning.className='explain';
      warning.textContent='Это старое предложение без действующего разрешения. Повтори запрос.';
      card.append(warning);
      return;
    }
    const panel = document.createElement('div');
    panel.className = 'web-permission';
    const label = document.createElement('p');
    label.textContent = `Джарвис предлагает поиск: «${permission.query}». Интернет пока НЕ использован.`;
    const approve = document.createElement('button');
    approve.type = 'button'; approve.textContent = '🔎 Разрешить этот поиск';
    approve.onclick = async () => {
      if (client.isBusy()) return;
      approve.disabled = true;
      try {
        await ensure();
        await client.api(`/api/sessions/${current}/events`, {
          method:'POST', body:JSON.stringify({
            event_type: 'web_search_consent',
            turn_id: originTurnId || null,
            data: {query: permission.query, question: permission.question}
          })
        });
        const question = permission.question || lastQuestion;
        client.putMessage(question, 'user', 'Повторный вопрос · поиск разрешён учителем');
        client.setBusy(true);
        await client.send(question, 'apprentice', {
          web_mode: 'ask', allow_web_search: true,
          approval_id: permission.id
        });
        panel.remove();
      } catch (error) {
        client.putMessage('Поиск не удался: ' + error.message, 'system');
        approve.disabled = false;
      } finally {
        client.setBusy(false);
      }
    };
    panel.append(label, approve); card.append(panel);
  }

  async function load(id) {
    const [record,changes] = await Promise.all([
      client.api('/api/sessions/' + encodeURIComponent(id)),
      client.api('/api/apprentice/changes')
    ]);
    const pendingChangeIds = new Set((changes.pending || []).map(x=>x.id));
    current = id;
    localStorage.setItem(storageKey, id);
    $('session-select').value = id;
    client.chat.replaceChildren();
    const history = [];
    lastQuestion = '';
    for (const turn of record.turns || []) {
      lastQuestion = turn.question;
      const info = [turn.route, turn.model, turn.total_ms != null ?
        `${turn.total_ms} мс` : '', turn.tokens != null ?
        `${turn.tokens} токенов` : '', turn.created_at + ' UTC']
        .filter(Boolean).join(' · ');
      client.putMessage(turn.question, 'user');
      const card = client.putMessage(turn.answer, 'assistant', info);
      if (turn.rating) {
        const badge = document.createElement('div');
        badge.className = 'msg-meta';
        badge.textContent = turn.rating === 'good' ?
          '✓ Учитель оценил ответ как верный' :
          '✕ Учитель отметил ошибку: ' + (turn.teacher_note || 'нет пояснения');
        card.append(badge);
      }
      addSources(card, turn.provenance?.sources);
      const proposal=turn.provenance?.change_proposal;
      if (proposal?.id && pendingChangeIds.has(proposal.id)){
        window.JarvisChatRenderChange?.(proposal,card,turn.id);
      }
      if (turn.provenance?.web_approval) {
        addPermission(card, turn.provenance.web_approval, turn.id);
      }
      if (turn.experiment === 'apprentice' || turn.experiment === 'hybrid') {
        history.push({role:'user',content:turn.question},
                     {role:'assistant',content:turn.answer});
      }
    }
    for (const ev of record.events || []) {
      if (ev.event_type === 'google_ai_import') {
        // External text stays in the audit log, never a teacher-confirmed fact.
        const text = ev.data?.text || '';
        client.putMessage('Google AI (вручную вставлено, не проверено): '+
                          text.slice(0, 400) + (text.length > 400 ? '…' : ''),
                          'system', ev.created_at + ' UTC');
      }
    }
    client.setHistory(history.slice(-14));
    if (!record.turns?.length) {
      client.putMessage('Новый исследовательский диалог. Сообщения и оценки будут сохранены в SQLite.',
                        'system');
    }
    alertStatus(`Сохранён в SQLite · ${record.turns?.length || 0} реплик · ${record.updated_at} UTC`);
  }

  async function newSession() {
    if (client.isBusy()) return;
    const created = await client.api('/api/sessions', {
      method:'POST',body:JSON.stringify({title:'Новый диалог'})
    });
    await list();
    await load(created.id);
    await list();
  }

  async function clearCurrentSession() {
    if (client.isBusy() || !current) return;
    const yes = window.confirm(
      'Удалить ВСЕ сообщения, оценки, события и трассировки из текущего диалога? '+
      'Это нельзя отменить. Подтверждённые знания и публичные находки '+
      'именованных пользователей сохраняются.'
    );
    if (!yes) return;
    const button = $('session-clear');
    button.disabled = true;
    try {
      const result = await client.api(
        '/api/sessions/' + encodeURIComponent(current) + '/clear',
        {method:'POST',body:JSON.stringify({confirm:true})});
      await load(current);
      await list();
      if ($('message')) $('message').value = '';
      if ($('google-question')) $('google-question').value = '';
      if ($('google-paste')) $('google-paste').value = '';
      alertStatus('Диалог очищен · удалено '+result.turns_deleted+
                  ' ответов и '+result.events_deleted+
                  ' событий. Подтверждённые знания сохранены.');
    } finally {
      button.disabled = false;
    }
  }

  async function download(format) {
    const id = await ensure();
    const last = format === 'md' ? 10 : 0;
    const url = `/api/sessions/${encodeURIComponent(id)}/export?format=${format}`+
                (last ? `&last=${last}` : '');
    const headers = {};
    if ($('token').value) headers['X-Jarvis-Token'] = $('token').value;
    const res = await fetch(url,{headers,cache:'no-store'});
    if (!res.ok) throw new Error(`Ошибка экспорта: HTTP ${res.status}`);
    const blob = await res.blob();
    const tempURL = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = tempURL;
    anchor.download = `jarvis-${id}${last ? '-last-'+last : ''}.${format}`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(tempURL), 5000);
  }

  async function onAnswer({card, response, question, mode}) {
    lastQuestion = question;
    if (!$('google-question').value.trim()) $('google-question').value = question;
    addSources(card, response.sources);
    if (mode === 'apprentice' && response.web_approval) {
      addPermission(card, response.web_approval, response.turn_id);
    }
    if (response.persistence_warning) {
      const warn = document.createElement('p');
      warn.className = 'save-warning';
      warn.textContent = '⚠️ Не сохранено в SQLite: ' + response.persistence_warning;
      card.append(warn);
      alertStatus('⚠️ Ошибка сохранения диалога');
    } else if (response.turn_id) {
      alertStatus(`Ответ #${response.turn_id} сохранён в SQLite`);
    }
    if (current) {
      try { await list(); }
      catch (error) { alertStatus('Ответ сохранён; список диалогов недоступен: ' + error.message); }
    }
  }

  async function onReview({entry, decision, turnId}) {
    if (!current) return;
    try {
      await client.api(`/api/sessions/${current}/events`, {
        method:'POST',body:JSON.stringify({
          event_type: 'knowledge_review',
          turn_id: turnId || null,
          data: {proposal_id:entry.id, kind:entry.kind, name:entry.name,
                 decision, teacher_details:entry.details || {}}
        })
      });
    } catch (error) {
      alertStatus('Предложение проверено, но запись в истории не удалась: '+error.message);
    }
  }

  async function onChangeReview({entry,result,turnId}){
    if(!current)return;
    try{
      await client.api(`/api/sessions/${current}/events`,{
        method:'POST',body:JSON.stringify({
          event_type:'knowledge_change_review',
          turn_id:turnId||null,
          data:{change_id:entry.id,target_kind:entry.target_kind,
                target_id:entry.target_id,
                before:entry.expected_value,after:result.proposed_value,
                status:result.status}
        })
      });
    }catch(error){
      alertStatus('Знание изменено, но запись события не удалась: '+error.message);
    }
  }

  async function grade(turnId, rating, teacherNote) {
    if (!current || !turnId) return;
    await client.api(`/api/sessions/${current}/grade`,{
      method:'POST',body:JSON.stringify({
        turn_id:turnId,rating,teacher_note:teacherNote||''
      })
    });
  }

  async function bootstrapSafe() {
    try { await bootstrap(); }
    catch (error) { alertStatus('История недоступна: '+error.message); }
  }

  async function init(handlers) {
    client = handlers;
    const previous = localStorage.getItem(webKey) || 'ask';
    if (['ask','auto','off'].includes(previous)) $('web-mode').value = previous;
    $('web-mode').addEventListener('change',()=>localStorage.setItem(webKey,$('web-mode').value));
    $('session-new').onclick = async () => {
      try { await newSession(); }
      catch(error) { alertStatus('Ошибка создания: '+error.message); }
    };
    $('session-clear').onclick = async () => {
      try { await clearCurrentSession(); }
      catch(error) { alertStatus('Ошибка очистки диалога: '+error.message); }
    };
    $('session-select').onchange = async event => {
      if (client.isBusy()) return;
      try { await load(event.target.value); }
      catch(error) { alertStatus('Ошибка загрузки: '+error.message); }
    };
    for (const format of ['md','json']) {
      $('download-'+format).onclick = async () => {
        try { await download(format); }
        catch(error) { alertStatus(error.message); }
      };
    }
    $('google-open').onclick = () => {
      const query = $('google-question').value.trim() ||
                    $('message').value.trim() || lastQuestion;
      if (!query) { alertStatus('Введи вопрос для Google AI.'); return; }
      $('google-question').value = query;
      // User gesture only; never automated Google scraping or login reuse.
      window.open('https://www.google.com/search?udm=50&q='+
                  encodeURIComponent(query),'_blank','noopener,noreferrer');
    };
    $('google-import').addEventListener('submit',async event => {
      event.preventDefault();
      if (client.isBusy()) return;
      const raw = $('google-paste').value.trim();
      const question = $('google-question').value.trim() ||
                       lastQuestion || 'Оцени приведённый внешний ответ';
      if (!raw) return;
      const button = $('google-import').querySelector('button[type=submit]');
      button.disabled = true;
      try {
        await ensure();
        await client.api(`/api/sessions/${current}/events`,{
          method:'POST',body:JSON.stringify({
            event_type:'google_ai_import',
            data:{question,text:raw.slice(0,8000),
                  provenance:'manual_paste_not_verified'}
          })
        });
        client.putMessage(question,'user','Анализ вручную вставленного Google AI ответа');
        client.setBusy(true);
        await client.send(question,'apprentice',{
          external_context:raw.slice(0,8000),web_mode:'off'
        });
        $('google-paste').value = '';
      } catch(error) {
        alertStatus('Ошибка Google AI импорта: '+error.message);
      } finally {
        client.setBusy(false);button.disabled=false;
      }
    });
    $('token').addEventListener('change',bootstrapSafe);
    boot = bootstrapSafe();
    await boot;
  }

  return {
    init, ready: ensure, id: () => current,
    webMode: () => $('web-mode').value,
    onAnswer, grade, onReview, onChangeReview
  };
})();
