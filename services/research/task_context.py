"""Actor-scoped task continuity derived from USER turns, never assistant guesses.

This is a read-only, bounded dialogue planner. It neither authorizes network
actions nor turns generated answers into facts. Explicit search intent and
omitted referents are resolved before an executor sees a query. A low-confidence
query must be clarified rather than sent to an external service.
"""
import re

from .normalize import normalize

# Tool vocabulary, not domain/topic vocabulary. Topics are taken from what
# the person actually said; neither recipes nor HA have special-case branches.
SEARCH_VERBS = ("найди", "найдите", "поищи", "поищите", "загугли")
OTHER_REQUESTS = ("дай", "дайте", "покажи", "покажите", "расскажи",
                  "расскажите", "объясни", "объясните", "подскажи")
FILLER = re.compile(
    r"(?iu)^(?:(?:мне\s+нуж(?:ен|на|но|ны)|"
    r"мені\s+потріб(?:ен|на|но|ні)|"
    r"я\s+хочу|мне|мені|нам|пожалуйста|будь\s+ласка|хочу|"
    r"можешь|можете|а|ну|слушай)\s+)+"
)
ONLINE = re.compile(
    r"(?iu)^(?:(?:в\s+интернете|в\s+сети|через\s+интернет|"
    r"онлайн|в\s+гугле|on\s+the\s+web)\s*)+"
)
TRAILING_ONLINE = re.compile(
    r"(?iu)\s+(?:в\s+интернете|в\s+сети|через\s+интернет|"
    r"онлайн|в\s+гугле|on\s+the\s+web)\s*$"
)
ACK = re.compile(
    r"(?iu)^\s*(?:да|нет|неа|ага|угу|так|ні|"
    r"подтверждаю|разрешаю(?:\s+поиск)?|да\s+разрешаю|"
    r"отменяю|согласен|ок|okay|yes|no)[!.\s]*$"
)
TEACHING = re.compile(
    r"(?iu)^\s*(?:когда|если)\s+(?:ты|вы|джарвис|jarvis)\b|"
    r"\b(?:означает|это\s+команда|значит\s+согласие)\b"
)
PERSONAL_QUESTION = re.compile(
    r"(?iu)^\s*(?:как|кто|что|где|когда|почему|сколько|"
    r"какой|какая|какое|какие|зачем|откуда)\b"
)
REFERENCE = re.compile(
    r"(?iu)^(?:это|эту|этот|эти|его|её|ее|их|"
    r"тот|ту|такой|такую|о\s+нем|о\s+ней|"
    r"по\s+нему|по\s+ней|об\s+этом|про\s+это)\b"
)
ADJECTIVE = re.compile(
    r"(?iu).{3,}(?:ый|ий|ой|ая|яя|ое|ее|ые|ие|"
    r"ого|его|ому|ему|ую|юю|ым|им|ими|ыми|"
    r"ій|а|е|і)$"
)


def one_edit(left, right):
    """Allow one ASR/typing edit only in command position, never in content."""
    if left == right:
        return True
    if len(left) < 5 or abs(len(left)-len(right)) > 1:
        return False
    if not left.startswith(right[:3]):
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    longer, shorter = (left, right) if len(left) > len(right) else (right, left)
    for i in range(len(longer)):
        if longer[:i] + longer[i+1:] == shorter:
            return True
    return False


def strip_exterior(text):
    value = str(text or "").strip(" \t\n.,!?;:—-")
    for _ in range(3):
        stripped=FILLER.sub("", value).strip(" \t\n.,!?;:—-")
        if stripped == value:
            break
        value=stripped
    value=ONLINE.sub("",value).strip(" \t\n.,!?;:—-")
    value=TRAILING_ONLINE.sub("",value).strip(" \t\n.,!?;:—-")
    return re.sub(r"\s+"," ",value).strip()


def search_clause(text):
    """Find an actual command clause, not a quoted lesson about commands.

    Supports verb-first requests and comma-separated clauses like
    "мне нужен X, поищи в интернете"; small ASR edit on the verb is OK.
    """
    if TEACHING.search(text):
        return None
    for index, clause in enumerate(re.split(r"[,;]", text)):
        clause = strip_exterior(clause)
        if not clause:
            continue
        tokens=clause.split(maxsplit=1)
        verb=normalize(tokens[0])
        if any(one_edit(verb, command) for command in SEARCH_VERBS):
            return index, tokens[1] if len(tokens)>1 else ""
        if re.match(r"(?iu)^проверь\s+(?:в\s+интернете|в\s+сети)\b", clause):
            return index, clause
        if re.match(r"(?iu)^search\s+(?:the\s+)?web\b", clause):
            return index, clause
    return None


def _words(text):
    return re.findall(r"(?u)[^\W_]+",normalize(text))


def _common(left,right):
    # Morphology-agnostic shared stem for inflected adjacent topic phrases.
    return left == right or (len(left)>=5 and len(right)>=5 and
                             left[:5]==right[:5])


def merge_focus(previous, current):
    """Fill an omitted referent using lexical evidence, without new nouns.

    'рецепт борща' + 'классический рецепт' -> 'классический рецепт борща'
    'рецепт котлет' + 'рецепт борща' -> 'рецепт борща' (new subject).
    """
    prev=str(previous or "").strip()
    curr=str(current or "").strip()
    if not curr:
        return prev, bool(prev)
    if not prev:
        return curr, False
    if REFERENCE.match(curr):
        return prev, True
    before=prev.split()
    after=curr.split()
    # Inspect the most specific (rightmost) shared head first. A second
    # clarification of "классический рецепт" must not discard "борща"
    # merely because "классический" also appeared in the previous focus.
    for ni in range(len(after)-1,-1,-1):
        new=after[ni]
        for oi in range(len(before)-1,-1,-1):
            old=before[oi]
            if not _common(normalize(new),normalize(old)):
                continue
            # A noun supplied after the shared head is an explicit new topic;
            # do not force its old complement into the current search.
            suffix=after[ni+1:]
            if suffix and not all(ADJECTIVE.fullmatch(token) for token in suffix):
                return curr, False
            # Descriptors (official, classical, newest) do not replace
            # the missing object of a previous noun phrase.
            missing=before[oi+1:]
            if missing and not suffix:
                return (curr+" "+" ".join(missing)).strip(), True
            return curr, False
    return curr, False


def _request_focus(text):
    raw = str(text or "").strip()
    if not raw or ACK.fullmatch(raw) or TEACHING.search(raw):
        return "", False, False
    search=search_clause(raw)
    if search is not None:
        index,tail=search
        if index>0:
            leading=re.split(r"[,;]",raw)[0]
            fragment=strip_exterior(leading)
        else:
            fragment=strip_exterior(tail)
        return fragment[:240], True, True
    clean=strip_exterior(raw)
    tokens=clean.split(maxsplit=1)
    if tokens and normalize(tokens[0]) in OTHER_REQUESTS:
        return strip_exterior(tokens[1] if len(tokens)>1 else "")[:240], False, True
    if PERSONAL_QUESTION.match(clean):
        # Questions can keep their own answer context in the LLM history,
        # but are not authoritative search subjects.
        return "", False, False
    # No command: do not promote an arbitrary assertion to a search topic.
    return "", False, False


def plan(previous_turns, request):
    """Derive working focus from the last bounded sequence of user requests.

    Only session+actor-scoped user turns are passed by the caller. Do not read
    assistant replies, old confirmed facts, or externally pasted snippets.
    """
    focus=""
    idle=0
    for turn in previous_turns[-12:]:
        item, _, addressed=_request_focus(turn)
        if addressed and item:
            focus,_=merge_focus(focus,item) if idle<=2 else (item,False)
            idle=0
        elif str(turn or "").strip() and not ACK.fullmatch(str(turn or "").strip()):
            idle+=1
            if idle>2:
                focus=""
    current, wants_search, addressed = _request_focus(request)
    if addressed:
        resolved, reused=merge_focus(focus,current) if idle<=2 else (current,False)
    else:
        resolved,reused=focus,False
    status=("ready" if not wants_search or resolved else "missing_subject")
    return {
        "subject":resolved[:240],
        "explicit_search":wants_search,
        "search_query":resolved[:240] if wants_search and status=="ready" else "",
        "context_used":bool(reused and addressed),
        "status":status,
        "source":"user_turns_session_actor",
        "history_turns_examined":min(len(previous_turns),12),
    }
