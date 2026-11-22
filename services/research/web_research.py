"""Free, narrowly scoped public-web research via a self-hosted SearXNG JSON API.

This is *not* arbitrary network.http: only the preconfigured SearXNG service
and verified public HTTPS search results are reachable. Web content is
untrusted evidence, never executable instructions or confirmed knowledge.
"""
import asyncio
import ipaddress
import os
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urlsplit

import httpx


class VisibleText(HTMLParser):
    SKIP={"script","style","noscript","svg","nav","footer","header","iframe",
          "button","form","aside","canvas"}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip=0
        self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag in self.SKIP:
            self.skip+=1
        elif tag in ("p","li","h1","h2","h3","br","article","section","td"):
            self.parts.append(" ")
    def handle_endtag(self,tag):
        if tag in self.SKIP and self.skip:
            self.skip-=1
        elif tag in ("p","li","h1","h2","h3","article","section"):
            self.parts.append("\n")
    def handle_data(self,data):
        if not self.skip and data.strip():
            self.parts.append(data)
    def content(self,limit=4000):
        value=" ".join(self.parts)
        value=re.sub(r"[ \t]+"," ",value)
        value=re.sub(r"\s*\n\s*","\n",value).strip()
        return value[:limit]


def public_https(url):
    """Reject private hosts, credentials, unusual ports and dangerous schemes."""
    if not isinstance(url,str) or len(url)>1600:
        return False
    try:
        parts=urlsplit(url.strip())
        host=parts.hostname
        try:
            port=parts.port
        except ValueError:
            return False
        if parts.scheme.lower()!="https" or not host or port not in (None,443):
            return False
        if parts.username is not None or parts.password is not None:
            return False
        host=host.encode("idna").decode("ascii").lower().rstrip(".")
        if host in ("localhost","localhost.localdomain") or host.endswith(".local"):
            return False
        ip=ipaddress.ip_address(host)
        return ip.is_global
    except ValueError:
        # A domain name is not an IP literal. DNS is separately validated.
        return bool(re.fullmatch(r"(?=.{4,253}$)[a-z0-9-]+(?:\.[a-z0-9-]+)+",host or "")) and not any(
            host.endswith(s) for s in (".internal",".localhost",".test",".invalid",".home",".lan")
        )
    except (UnicodeError,TypeError):
        return False


def public_addresses(host):
    """DNS-based SSRF check; redirects are prohibited as additional protection."""
    try:
        addresses=socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)
    except (socket.gaierror,TimeoutError,OSError):
        return False
    return bool(addresses) and all(ipaddress.ip_address(info[4][0]).is_global for info in addresses)


async def checked_url(url):
    if not public_https(url):
        return False
    host=urlsplit(url).hostname
    return await asyncio.to_thread(public_addresses,host)


class FreeWebResearch:
    def __init__(self,engine):
        self.engine=engine
        with engine.connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS research_web_searches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT '',
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    results INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_web_search_rate
                ON research_web_searches(session_id,created_at);
            """)

    def _rate_ok(self,sid):
        with self.engine.connect() as con:
            count=con.execute("""
                SELECT COUNT(*) FROM research_web_searches
                WHERE session_id=? AND created_at >= datetime('now','-10 minutes')
            """,(sid,)).fetchone()[0]
            return count<8

    def _log(self,sid,query,status,count):
        with self.engine.connect() as con:
            con.execute("""INSERT INTO research_web_searches(session_id,query,status,results)
                           VALUES(?,?,?,?)""",(sid,query[:280],status,count))

    async def _page(self,url):
        if not await checked_url(url):
            return "", "blocked_public_https_only"
        try:
            # Strip proxy environment to prevent accidental access to an
            # administrator-supplied internal proxy through untrusted URLs.
            timeout=httpx.Timeout(9.0,connect=4.0)
            async with httpx.AsyncClient(
                timeout=timeout,follow_redirects=False,trust_env=False,
                headers={"User-Agent":"JARVIS-Research/0.5 (public read only)"}
            ) as client:
                async with client.stream("GET",url) as response:
                    if response.status_code in (301,302,303,307,308):
                        return "", "redirect_not_followed"
                    if response.status_code!=200:
                        return "", f"http_{response.status_code}"
                    if "text/html" not in response.headers.get("content-type","").lower():
                        return "", "unsupported_content_type"
                    chunks=[]
                    size=0
                    async for chunk in response.aiter_bytes():
                        size+=len(chunk)
                        if size>240000: break
                        chunks.append(chunk)
            parser=VisibleText()
            parser.feed(b"".join(chunks).decode("utf-8",errors="replace"))
            content=parser.content()
            return content,("read" if content else "empty_page")
        except (httpx.HTTPError,UnicodeError,ValueError,OSError):
            return "", "fetch_failed"

    async def search(self,query,session_id=""):
        enabled=os.getenv("WEB_SEARCH_ENABLED","true").casefold() in ("1","true","yes")
        if not enabled:
            return {"available":False,"query":query,"status":"disabled","sources":[],
                    "reason":"Поиск отключён администратором"}
        query=re.sub(r"\s+"," ",str(query)).strip()[:240]
        if len(query)<3:
            return {"available":False,"query":query,"status":"invalid_query",
                    "sources":[],"reason":"Слишком короткий поисковый запрос"}
        sid=str(session_id or "")[:60]
        if not self._rate_ok(sid):
            return {"available":False,"query":query,"status":"rate_limited",
                    "sources":[],"reason":"Лимит: 8 поисков за 10 минут на диалог"}
        base=os.getenv("SEARXNG_URL","http://searxng:8080").rstrip("/")
        # Only preconfigured in-container search; no caller-supplied base URL.
        status="unavailable"
        sources=[]
        try:
            async with httpx.AsyncClient(timeout=14,trust_env=False) as client:
                response=await client.get(base+"/search",params={
                    "q":query,"format":"json","categories":"general","language":"all"})
                response.raise_for_status()
                payload=response.json()
            seen=set()
            for raw in payload.get("results",[])[:20]:
                if not isinstance(raw,dict):
                    continue
                url=str(raw.get("url") or "").strip()
                if not public_https(url) or url in seen:
                    continue
                seen.add(url)
                title=re.sub(r"\s+"," ",str(raw.get("title") or "Источник")).strip()[:180]
                snippet=re.sub(r"\s+"," ",str(raw.get("content") or "")).strip()[:600]
                sources.append({"title":title,"url":url,"snippet":snippet,
                                "page_text":"","status":"search_snippet"})
                if len(sources)>=4:break
            # Read just two URLs; other results retain clearly labeled snippets.
            reads=await asyncio.gather(*(self._page(s["url"]) for s in sources[:2]))
            for source,(content,reason) in zip(sources,reads):
                source["page_text"]=content
                source["status"]=reason
            status="ok" if sources else "no_results"
        except (httpx.HTTPError,ValueError,KeyError,TypeError):
            status="searxng_unavailable"
        self._log(sid,query,status,len(sources))
        return {"available":status=="ok","query":query,"status":status,
                "sources":sources,
                "reason":"" if status=="ok" else {
                    "no_results":"Поисковик не вернул подходящих публичных HTTPS-источников",
                    "searxng_unavailable":"SearXNG недоступен или не возвращает JSON"
                }.get(status,"Поиск недоступен"),
                "tool":"public_web.search", "executed":True}
