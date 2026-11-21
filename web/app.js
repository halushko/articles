'use strict';
const $ = id => document.getElementById(id);
const chat = $('chat');
let targets = [];
let busy = false;

async function api(path, opts = {}) {
  const headers = {'Content-Type': 'application/json'};
  if ($('token').value) headers['X-Jarvis-Token'] = $('token').value;
  const response = await fetch(path, {...opts, headers});
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  return data;
}
function putMessage(text, role='assistant', meta='') {
  const div = document.createElement('div'); div.className = 'message '+role;
  const body = document.createElement('div'); body.textContent = text;
  div.appendChild(body);
  if (meta) { const foot = document.createElement('div'); foot.className='msg-meta';foot.textContent = meta;div.appendChild(foot); }
  chat.appendChild(div); chat.scrollTop=chat.scrollHeight;
  return div;
}
function setupCorrection(container, phrase) {
  const panel = document.createElement('div');panel.className='choice';
  const intro = document.createElement('div');intro.textContent='Какое демо-устройство ты имел в виду?'; panel.appendChild(intro);
  const selection = document.createElement('select');
  for (const target of targets) {const option=document.createElement('option');option.value=target.id;option.textContent=target.name+' ('+target.id+')';selection.appendChild(option);}
  panel.appendChild(selection);
  const save = document.createElement('button');save.textContent='Подтвердить и обучить';
  save.onclick=async () => {
    save.disabled=true;
    try {
      const res=await api('/api/learn',{method:'POST',body:JSON.stringify({phrase,target_id:selection.value})});
      putMessage(`Сохранил: ${res.phrase_key} → ${res.current_target}. Старых подтверждений: ${res.history_entries_for_phrase}. Физических действий: 0.`, 'assistant', 'research · teach');
      panel.remove();await refresh();
    } catch(e) {putMessage('Ошибка обучения: '+e.message,'system');save.disabled=false;}
  };
  panel.appendChild(save); container.appendChild(panel);chat.scrollTop=chat.scrollHeight;
}
async function send(text, experiment) {
  const opts={text,experiment:experiment||$('experiment').value,profile:$('profile').value,ablation:$('ablation').value};
  const response = await api('/api/chat',{method:'POST',body:JSON.stringify(opts)});
  const stamp=[response.route,response.model||'',`${response.total_ms} мс`,response.tokens!=null?`${response.tokens} токенов`:null].filter(Boolean).join(' · ');
  const card=putMessage(response.answer,'assistant',stamp);
  if (response.is_command && response.status === 'needs_confirmation' && experiment !== 'llm_only') setupCorrection(card,text);
  return response;
}
$('chat-form').addEventListener('submit',async event=>{
  event.preventDefault();if(busy)return;
  const message=$('message');const text=message.value.trim();if(!text)return;
  message.value='';putMessage(text,'user');busy=true;$('send').disabled=true;
  try{await send(text)}catch(e){putMessage('Ошибка: '+e.message,'system')}
  finally{busy=false;$('send').disabled=false}
});
$('run-ab').addEventListener('click',async()=>{
  const text=$('message').value.trim();if(!text){putMessage('Напиши запрос в поле ввода — затем нажми A/B.','system');return;}
  if(busy)return;busy=true;$('run-ab').disabled=true;
  putMessage('A/B: '+text,'user');
  try{putMessage('BASELINE · только LLM','system');await send(text,'llm_only');putMessage('HYBRID · индекс → LLM','system');await send(text,'hybrid');}
  catch(e){putMessage('A/B ошибка: '+e.message,'system')}
  finally{busy=false;$('run-ab').disabled=false}
});
async function refresh(){
  try {
    const [mem,health,result]=await Promise.all([api('/api/memory'),api('/api/health'),api('/api/targets')]);targets=result;
    const stats=$('stats');stats.replaceChildren();
    const names={corrections:'Исправлений',active_mappings:'Актуальных фраз',weight_features:'Весов признаков',concepts:'Понятий',semantic_vectors:'Векторов',decisions:'Решений'};
    for(const [key,label] of Object.entries(names)){
      const el=document.createElement('div');el.className='stat';const strong=document.createElement('strong');strong.textContent=mem.counts[key];const small=document.createElement('small');small.textContent=label;el.append(strong,small);stats.append(el);
    }
    const recent=$('recent');recent.replaceChildren();
    for(const r of mem.recent_corrections.slice(0,5)) {const entry=document.createElement('div');entry.textContent=`${r.phrase} → ${r.target_id}`;recent.appendChild(entry);}
    $('health').textContent=`Index: ${health.research?.status||'?'} · LLM: ${health.llm?.status||'?'}`;
  } catch(e){$('health').textContent='Нет связи: '+e.message;}
}
$('refresh').onclick=refresh;
$('show-memory').onclick=async()=>{try {const data=await api('/api/memory');putMessage(JSON.stringify(data,null,2),'assistant','SQLite · direct read')}catch(e){putMessage('Ошибка памяти: '+e.message,'system')}};
$('concept-form').addEventListener('submit',async event=>{
  event.preventDefault();
  try{
    const payload={name:$('concept-name').value,domain:$('concept-domain').value,kind:$('concept-kind').value};
    const res=await api('/api/concepts',{method:'POST',body:JSON.stringify(payload)});
    $('concept-status').textContent=`Добавлено понятие #${res.id}: ${res.name}`;$('concept-name').value='';await refresh();
  }catch(e){$('concept-status').textContent='Ошибка: '+e.message;}
});
refresh();
