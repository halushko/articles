'use strict';
const $ = id => document.getElementById(id);
const chat = $('chat');
let targets = [], busy = false, turnHistory = [];
const kindLabels = {entity:'Сущность',action:'Описание действия',fact:'Связь / факт',
                    lesson:'Урок / алгоритм',capability:'Гипотеза навыка'};

async function api(path, opts = {}) {
  const headers = {'Content-Type':'application/json'};
  if ($('token').value) headers['X-Jarvis-Token'] = $('token').value;
  const response = await fetch(path,{...opts,headers});
  const data = await response.json();
  if (!response.ok) {
    const error = typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||data);
    throw new Error(error||`HTTP ${response.status}`);
  }
  return data;
}
function putMessage(text, role='assistant', meta='') {
  const box=document.createElement('div');box.className='message '+role;
  const body=document.createElement('div');body.textContent=text;box.append(body);
  if(meta){const foot=document.createElement('div');foot.className='msg-meta';foot.textContent=meta;box.append(foot);}
  chat.append(box);chat.scrollTop=chat.scrollHeight;return box;
}
function mkButton(text,cls,cb) {
  const button=document.createElement('button');
  button.textContent=text;button.type='button';button.className=cls;button.onclick=cb;
  return button;
}
function renderProposal(entry, target, description=entry.description||'') {
  const panel=document.createElement('div');panel.className='proposal';
  const kind=document.createElement('div');kind.className='proposal-kind';
  kind.textContent=kindLabels[entry.kind]||entry.kind;
  const name=document.createElement('strong');name.textContent=entry.name;
  const detail=document.createElement('p');detail.textContent=description;
  const actions=document.createElement('div');actions.className='review-actions';
  const confirm=mkButton('✓ Подтвердить','',()=>review('confirm'));
  const reject=mkButton('✕ Отклонить','secondary',()=>review('reject'));
  const edit=mkButton('Изменить','secondary',()=>{
    const newName=window.prompt('Уточни название',name.textContent);
    if(newName===null)return;
    const newDescription=window.prompt('Уточни смысл записи',detail.textContent);
    if(newDescription===null)return;
    if(!newName.trim()||!newDescription.trim())return;
    name.textContent=newName.trim();detail.textContent=newDescription.trim();
  });
  async function review(decision){
    for(const b of [confirm,reject,edit])b.disabled=true;
    try{
      const result=await api('/api/apprentice/review',
        {method:'POST',body:JSON.stringify({proposal_id:entry.id,decision,
                        name:name.textContent,description:detail.textContent})});
      panel.remove();
      putMessage(decision==='confirm'?
        `Подтверждено: ${result.name} · ${kindLabels[result.kind]||result.kind}. Сохранил в память, но никаких действий не выполнял.`:
        `Отклонено: ${result.name}. Не использую как знание.`,'system','teacher review');
      await refresh();
    }catch(e){
      putMessage('Ошибка подтверждения: '+e.message,'system');
      for(const b of [confirm,reject,edit])b.disabled=false;
    }
  }
  actions.append(confirm,reject,edit);panel.append(kind,name,detail,actions);target.append(panel);
  chat.scrollTop=chat.scrollHeight;
}
function setupCorrection(container,phrase) {
  const panel=document.createElement('div');panel.className='choice';
  const intro=document.createElement('div');intro.textContent='Выбери демо-устройство из старого индекса:';panel.append(intro);
  const select=document.createElement('select');
  for(const target of targets) {
    const opt=document.createElement('option');opt.value=target.id;
    opt.textContent=`${target.name} (${target.id})`;select.append(opt);
  }
  panel.append(select);
  const save=mkButton('Обучить старый индекс','',async()=>{
    save.disabled=true;
    try{
      const r=await api('/api/learn',{method:'POST',
        body:JSON.stringify({phrase,target_id:select.value})});
      putMessage(`Сохранено: ${r.phrase_key} → ${r.current_target}`,'system');
      panel.remove();await refresh();
    }catch(e){putMessage('Ошибка: '+e.message,'system');save.disabled=false;}
  });
  panel.append(save);container.append(panel);
}
async function send(text,experiment) {
  const mode=experiment||$('experiment').value;
  const options={text,experiment:mode,profile:$('profile').value,
                 ablation:$('ablation').value,
                 history:mode==='apprentice'?turnHistory.slice(-12):[]};
  const r=await api('/api/chat',{method:'POST',body:JSON.stringify(options)});
  const stamp=[r.route,r.model,Number.isFinite(r.total_ms)?`${r.total_ms} мс`:null,
               r.tokens!=null?`${r.tokens} токенов`:null].filter(Boolean).join(' · ');
  if(r.normalized_input && r.normalized_input!==text) {
    putMessage('Словарь произношений: «'+text+'» → «'+r.normalized_input+'»','system');
  }
  const card=putMessage(r.answer,'assistant',stamp);
  const replay=mkButton('🔊 Повторить','replay-button',()=>voice.speak(r.answer,true));
  replay.setAttribute('aria-label','Повторить ответ голосом');
  card.append(replay);
  voice.speak(r.answer);
  if(r.parse_warning) putMessage('Не удалось надёжно извлечь структурированное знание. Ничего не добавлено.','system');
  if(mode==='apprentice'){
    const gradeControls=document.createElement('div');gradeControls.className='grade-controls';
    const verdict=document.createElement('span');verdict.textContent='Проверка учителя:';
    const buttons=[];
    async function grade(rating){
      let teacherNote='';
      if(rating==='bad'){
        const note=window.prompt('Что неправильно? Напиши исправление для будущего урока (необязательно):','');
        if(note===null)return;
        teacherNote=note;
      }
      buttons.forEach(x=>x.disabled=true);
      try{
        await api('/api/apprentice/grade',{method:'POST',
           body:JSON.stringify({question:text,answer:r.answer,rating,teacher_note:teacherNote})});
        gradeControls.textContent=rating==='good'?'✓ Ответ оценён как верный':'✕ Ошибка учтена в статистике. Объясни правильный ответ следующим сообщением, чтобы обучить память.';
        await refresh();
      }catch(e){
        putMessage('Ошибка оценки: '+e.message,'system');
        buttons.forEach(x=>x.disabled=false);
      }
    }
    const good=mkButton('✓ Верно','secondary',()=>grade('good'));
    const bad=mkButton('✕ Ошибка','secondary',()=>grade('bad'));
    buttons.push(good,bad);gradeControls.append(verdict,good,bad);card.append(gradeControls);
  }
  if(mode==='apprentice'){
    turnHistory.push({role:'user',content:r.normalized_input||text},{role:'assistant',content:r.answer});
    turnHistory=turnHistory.slice(-14);
    if(r.proposals?.length){
      const title=document.createElement('div');title.className='proposal-caption';
      title.textContent='Я заметил новые знания / возможности. Проверь:';
      card.append(title);
      for(const p of r.proposals)renderProposal(p,card);
    }
    if(r.missing){
      const missing=document.createElement('div');missing.className='msg-meta';
      missing.textContent='Не хватает: '+r.missing;card.append(missing);
    }
    await refresh();
  }else if(r.is_command && r.status==='needs_confirmation' && mode!=='llm_only'){
    setupCorrection(card,text);
  }
  return r;
}
// Browser speech is implemented by a testable voice controller.
const voiceButton = $('voice');
const messageInput = $('message');
const chatForm = $('chat-form');
// Browser textarea: Enter sends, Shift+Enter inserts a real newline.
messageInput.addEventListener('keydown', event => {
  if (event.key !== 'Enter' || event.shiftKey || event.altKey ||
      event.isComposing || event.keyCode === 229) return;
  event.preventDefault();
  if (!busy && messageInput.value.trim()) chatForm.requestSubmit();
});
const voice = window.JarvisVoice.initVoice({
  input: messageInput, button: voiceButton,
  language: $('voice-language'), status: $('voice-status'),
  form: chatForm, outputToggle: $('voice-output'),
  outputStatus: $('tts-status'), isBusy: () => busy
});
$('chat-form').addEventListener('submit',async event=>{
  event.preventDefault();if(busy)return;
  const input=$('message'),text=input.value.trim();if(!text)return;
  voice.stopForManualSend();
  voice.cancelSpeech();
  input.value='';putMessage(text,'user');busy=true;$('send').disabled=true;
  voiceButton.disabled=true;
  try{await send(text);}catch(e){putMessage('Ошибка: '+e.message,'system');}
  finally{
    busy=false;$('send').disabled=false;
    voiceButton.disabled=!voice.isSupported;input.focus();
  }
});
$('experiment').addEventListener('change',()=>{turnHistory=[];});
$('speech-form').addEventListener('submit',async event=>{
  event.preventDefault();
  const heard=$('speech-heard').value.trim(),correct=$('speech-correct').value.trim();
  if(!heard || !correct)return;
  const submit=$('speech-form').querySelector('button[type="submit"]');
  submit.disabled=true;
  try{
    const result=await api('/api/apprentice/speech-corrections',
      {method:'POST',body:JSON.stringify({heard,correct})});
    putMessage('Изучил произношение: «'+result.heard+'» → «'+result.correct+'».','system');
    $('speech-heard').value='';$('speech-correct').value='';
    await refresh();
  }catch(e){putMessage('Ошибка обучения произношению: '+e.message,'system');}
  finally{submit.disabled=false;}
});
document.querySelectorAll('[data-example]').forEach(button=>{
  button.addEventListener('click',()=>{$('message').value=button.dataset.example;$('message').focus();});
});
$('run-ab').onclick=async()=>{
  const text=$('message').value.trim();
  if(!text){putMessage('Сначала введи фразу для сравнения.','system');return;}
  if(busy)return;busy=true;$('run-ab').disabled=true;
  putMessage('Сравнение: '+text,'user');
  try{
    putMessage('LLM baseline','system');await send(text,'llm_only');
    putMessage('Legacy hybrid','system');await send(text,'hybrid');
  }catch(e){putMessage('Ошибка сравнения: '+e.message,'system');}
  finally{busy=false;$('run-ab').disabled=false;}
};
async function refresh(){
  try{
    const [memory,health,existing]=await Promise.all([
      api('/api/apprentice/state'),api('/api/health'),api('/api/targets')]);
    targets=existing;
    $('health').textContent=`Research: ${health.research?.status||'?'} · Qwen: ${health.llm?.status||'?'}`;
    const counts=memory.counts;
    const grid=$('learning-stats');grid.replaceChildren();
    const labels={entity:'Сущности',action:'Действия (опис.)',fact:'Связи',
                  lesson:'Уроки',capability:'Идеи навыков',pending:'Гипотезы',graded:'Оценено ответов'};
    for(const [key,label] of Object.entries(labels)){
      const box=document.createElement('div');box.className='stat';
      const strong=document.createElement('strong');strong.textContent=counts[key]||0;
      const small=document.createElement('small');small.textContent=label;
      box.append(strong,small);grid.append(box);
    }
    const known=$('known');known.replaceChildren();
    let any=false;
    for(const [kind,list] of Object.entries(memory.knowledge)){
      for(const item of list.slice(0,5)){
        any=true;const row=document.createElement('div');row.className='known-row';
        const name=document.createElement('strong');name.textContent=item.name||item.title||item.subject;
        const type=document.createElement('span');type.textContent=kindLabels[kind]||kind;
        const forget=mkButton('Забыть','text-button',async()=>{
          if(!window.confirm(`Отозвать подтверждение: ${name.textContent}?`))return;
          forget.disabled=true;
          try{
            await api('/api/apprentice/forget',{method:'POST',
              body:JSON.stringify({kind,item_id:item.id})});
            putMessage(`Знание отозвано: ${name.textContent}. Запись осталась в журнале.`,'system');
            await refresh();
          }catch(e){putMessage('Ошибка: '+e.message,'system');forget.disabled=false;}
        });
        row.append(name,type,forget);known.append(row);
      }
    }
    if(!any)known.textContent='Пока пусто. Начни с объяснения.';
    const dialogue=memory.dialogue||{};
    const roles=dialogue.roles||{};
    const identity=$('dialogue-roles');
    identity.replaceChildren();
    for(const [key,name] of [
      ['JARVIS',roles.assistant||'JARVIS'],
      ['Собеседник',roles.user||'имя пока неизвестно']
    ]){
      const row=document.createElement('div');row.className='known-row';
      const left=document.createElement('span');left.textContent=key;
      const right=document.createElement('strong');right.textContent=name;
      row.append(left,right);identity.append(row);
    }
    const dictionary=$('speech-items');dictionary.replaceChildren();
    for(const item of dialogue.speech_corrections||[]){
      const row=document.createElement('div');row.className='speech-row';
      const words=document.createElement('span');
      words.textContent=item.heard+' → '+item.correct;
      const del=mkButton('Удалить','text-button',async()=>{
        del.disabled=true;
        try{
          await api('/api/apprentice/speech-corrections/delete',
            {method:'POST',body:JSON.stringify({item_id:item.id})});
          await refresh();
        }catch(e){putMessage('Ошибка удаления: '+e.message,'system');del.disabled=false;}
      });
      row.append(words,del);dictionary.append(row);
    }
    if(!dialogue.speech_corrections?.length)
      dictionary.textContent='Словарь пока пуст.';
    const register=$('tool-register');register.replaceChildren();
    const statusLabels={available:'Доступно',teacher_only:'Учитель',unavailable:'Нет доступа'};
    for(const tool of dialogue.capabilities||[]){
      const row=document.createElement('div');row.className='tool-row';
      const details=document.createElement('div');
      const title=document.createElement('strong');title.textContent=tool.name;
      const info=document.createElement('small');info.textContent=tool.description;
      details.append(title,info);
      const badge=document.createElement('span');
      badge.className='tool-status '+tool.status;
      badge.textContent=statusLabels[tool.status]||tool.status;
      row.append(details,badge);register.append(row);
    }
    $('pending-count').textContent=String(counts.pending||0);
    const pending=$('pending');pending.replaceChildren();
    for(const entry of memory.pending.slice(0,15))renderProposal(entry,pending);
    if(!memory.pending.length){
      const empty=document.createElement('p');empty.className='explain';
      empty.textContent='Все предложения проверены.';pending.append(empty);
    }
  }catch(e){$('health').textContent='Нет связи: '+e.message;}
}
$('refresh').onclick=refresh;
$('show-learning').onclick=async()=>{
  try{
    const state=await api('/api/apprentice/state');
    putMessage(JSON.stringify(state.knowledge,null,2),'assistant','Подтверждённые знания · SQLite');
  }catch(e){putMessage('Ошибка: '+e.message,'system');}
};
$('show-memory').onclick=async()=>{
  try{
    const mem=await api('/api/memory');
    putMessage(JSON.stringify(mem,null,2),'assistant','Legacy research index');
  }catch(e){putMessage('Ошибка: '+e.message,'system');}
};
$('token').addEventListener('change',refresh);
refresh();
