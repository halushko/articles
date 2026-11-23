"""One read model over JARVIS' existing SQLite world, identity and experience.

No second knowledge database: confirmed world_entities/facts/graph remain the
authority; journalled user statements remain episodes, NEVER confirmed facts.
Neither Qwen output nor an unreviewed statement becomes a graph assertion.
"""
from .normalize import normalize
from .utterance import classify

# Function words, not domain-specific canned answers.
STOP = frozenset(
    "я ты он она мы вы они мне меня тебе тебя мой моя мое моё мои твой твоя "
    "это этот эта эти что как где когда кто почему зачем какой какие "
    "для при или про уже ещё нужно надо можно хочу скажи расскажи "
    "у в на и а но не из по от с со за к ко о об до вон "
    "і й у в на що як хто де коли чому цей це моя мої "
    "the a an and or to for what why how who when where"
    .split()
)


def terms(text):
    return {t for t in normalize(str(text)).split()
            if len(t) >= 3 and t not in STOP}


class WorldMemory:
    def __init__(self, engine, apprentice, dialogue, behavior):
        self.engine = engine
        self.apprentice = apprentice
        self.dialogue = dialogue
        self.behavior = behavior

    def recall(self, query, actor_id="", session_id="", limit=4):
        """Retrieve cross-session *user statements*, scoped to one actor.

        Assistant replies are intentionally excluded: a model's mistake must
        not become a new autobiographical fact through retrieval.
        Unknown speakers cannot cross session boundaries.
        """
        if not actor_id:
            return []
        query_terms=terms(query)
        if not query_terms:
            return []
        with self.engine.connect() as con:
            rows=con.execute("""SELECT id,session_id,question,created_at
                 FROM research_turns WHERE actor_id=? AND session_id<>?
                 ORDER BY id DESC LIMIT 500""",(actor_id,session_id or "")).fetchall()
        ranked=[]
        for index, row in enumerate(rows):
            utterance=str(row["question"] or "").strip()
            # Questions and commands are not autobiographical observations.
            if classify(utterance)["intent"] != "statement":
                continue
            words=terms(utterance)
            overlap=query_terms.intersection(words)
            if not overlap:
                continue
            score=len(overlap)/max(1,len(query_terms))
            score+=min(0.08,0.08/(1+index/100))
            if score < 0.23:
                continue
            ranked.append((score,row))
        ranked.sort(key=lambda pair:(-pair[0],-pair[1]["id"]))
        return [{
            "turn_id":row["id"], "session_id":row["session_id"],
            "created_at":row["created_at"],
            "text":row["question"][:600],
            "relevance":round(score,3),
            "source":"user_statement_in_dialogue",
            "verified":False,
            "note":"Эпизод разговора, НЕ подтверждённый факт о мире."
        } for score,row in ranked[:max(1,min(limit,6))]]

    def context(self, query, speaker=None, actor_id="", session_id=""):
        data=self.apprentice.context(query,speaker=speaker)
        data["behavior_rules"]=self.behavior.context(actor_id)
        data["dialogue"]=self.dialogue.context(session_id,actor_id)
        data["task_context"]=self.dialogue.task_context(session_id,actor_id,
                                                            query)
        data["operations"]=(self.dialogue.approvals.execution_status(session_id,actor_id)
                            if session_id else {"current":None,"last_result":None})
        data["speaker"]=speaker
        data["experiences"]=self.recall(data.get("normalized_text") or query,
                                        actor_id,session_id)
        data["world_model"]={
            "confirmed_source":"SQLite world_entities/world_facts/graph_nodes",
            "episodic_source":"research_turns.question from selected speaker",
            "confirmed_knowledge_only":True,
            "episodes_are_unverified":True,
            "assistant_messages_are_not_world_facts":True,
            "actor_profile_is_not_authentication":True,
        }
        return data
