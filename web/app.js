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
                 history:mode==='apprentice'?turnHistory.slice(-6):[]};
  const r=await api('/api/chat',{method:'POST',body:JSON.stringify(options)});
  const stamp=[r.route,r.model,Number.isFinite(r.total_ms)?`${r.total_ms} мс`:null,
               r.tokens!=null?`${r.tokens} токенов`:null].filter(Boolean).join(' · ');
  const card=putMessage(r.answer,'assistant',stamp);
  if(r.parse_warning) putMessage('Не удалось надёжно извлечь структурированное знание. Ничего не добавлено.','system');
  if(mode==='apprentice'){
    turnHistory.push({role:'user',content:text},{role:'assistant',content:r.answer});
    turnHistory=turnHistory.slice(-8);
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
$('chat-form').addEventListener('submit',async event=>{
  event.preventDefault();if(busy)return;
  const input=$('message'),text=input.value.trim();if(!text)return;
  input.value='';putMessage(text,'user');busy=true;$('send').disabled=true;
  try{await send(text);}catch(e){putMessage('Ошибка: '+e.message,'system');}
  finally{busy=false;$('send').disabled=false;input.focus();}
});
$('experiment').addEventListener('change',()=>{turnHistory=[];});
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
                  lesson:'Уроки',capability:'Идеи навыков',pending:'Гипотезы'};
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
        row.append(name,type);known.append(row);
      }
    }
    if(!any)known.textContent='Пока пусто. Начни с объяснения.';
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
