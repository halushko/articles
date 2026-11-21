# JARVIS v0.3.2 — отделённая LLM и исследовательская индексация

Разработка ведётся **в ветке `llm-index` репозитория `halushko/articles`**. `master` не трогаем.

## Принцип разработки и развёртывания

Любое изменение коммитится в `llm-index` → GitHub Actions прогоняет тесты → публикует три контейнера в GHCR. **На Windows не нужен исходный код Python, локальная сборка, `git pull` или копирование файлов**. После зелёного workflow достаточно в папке с неизменным `compose.yaml` выполнить:

```powershell
docker compose down
docker compose pull
docker compose up -d
```

`down` **не** удаляет volumes. Не используй `down -v`: он удалит индекс и скачанные модели. `pull` скачивает обновлённые образы, `up -d` запускает их. После коммита **подожди завершения публикации всех трёх образов**: если сделать `pull` во время workflow, можно получить разные версии сервисов.

**Важный нюанс:** изменения в `compose.yaml` и `.env.example` не подтягиваются командой `docker compose pull`: это локальные файлы. Мы считаем их стабильным контрактом, изменения алгоритмов и настроек LLM/Research храним в образах. Если когда-нибудь придётся менять Compose, это отдельное миграционное обновление, а не обычный down-pull-up.

## Первый запуск (один раз)

```powershell
git clone --branch llm-index --single-branch https://github.com/halushko/articles.git jarvis-llm-index
cd jarvis-llm-index
Copy-Item .env.example .env
docker compose pull
docker compose up -d
```

Открой `http://localhost:8766`. Старый v0.3.1 на порту 8765 не трогаем.

При первом запуске `model-init` **сам скачает** Qwen2.5:3b и Qwen3:4b в отдельный постоянный том. На следующих запусках уже скачанные модели не перекачиваются. Операция `docker compose up -d` может ожидать первоначальную загрузку. Для проверки:

```powershell
docker compose ps
docker compose logs --tail=60 gateway llm research model-init
```

Образы GHCR становятся доступны после успешного GitHub Actions workflow. Если возникнет `denied`, пакеты пока приватны — один раз открой настройки соответствующего GHCR package и включи Public или выполни `docker login ghcr.io` с токеном на чтение пакетов. **Ни пароли, ни токены не добавляй в репозиторий.**

## Независимые части

| Контейнер | Образ GHCR | Что меняем в GitHub | Данные |
|---|---|---|---|
| `llm` | `articles-llm-index-llm` | `services/llm/`, `config/llm.json` | Нет доступа к БД |
| `research` | `articles-llm-index-research` | `services/research/`, `config/research.json` | `research_data` (SQLite) |
| `gateway` | `articles-llm-index-gateway` | `services/gateway/`, `web/` | Нет |
| `ollama` | `ollama/ollama:0.40.17` | Закреплённая версия готового движка | `ollama_models` |

У контейнеров LLM и Research **разные Docker-образы, API и права доступа к данным**. Research не импортирует сервис чата, LLM не содержит исследовательского кода и базы.

`config/llm.json` задаёт профили `fast`/`deep`, модели, thinking, температуру и лимиты. `config/research.json` задаёт learning rate, пороги, допустимые ранжировщики и семантические параметры. **Меняем в репозитории, затем down-pull-up на локальной машине**, а не редактируем локальные конфиги.

Режимы через UI: `llm_only` (базовая LLM), `research_only` (свой индекс без чата), `hybrid` (индекс для известных команд, LLM для остального). Абляции индекса: `exact`, `online`, `combined` (опциональная замороженная BGE-M3). Смена режима в UI не требует перезапуска.

## API и исследование

```text
GET  /api/health
GET  /api/memory
GET  /api/targets
POST /api/chat {"text":"...", "experiment":"hybrid|research_only|llm_only", "profile":"fast|deep", "ablation":"exact|online|combined"}
POST /api/learn {"phrase":"Включи свет в туалете", "target_id":"light.lab_toilet"}
POST /api/concepts {"name":"Духовка", "domain":"техника", "kind":"устройство"}
POST /api/feedback {"decision_id":1, "rating":"good|bad"}
```

Публичный API только через `gateway` (localhost:8766), остальные контейнеры в приватной сети Compose. Реальные устройства Home Assistant **не управляются**, возвращается `executed: false`. Индекс и LLM независимы от старой v0.3.1. Голос и TTS не включены.

Для разработчика: тесты CI — `python -m unittest discover -s tests -v`, затем `python -m evaluation.benchmark --dataset evaluation/toy_cases.jsonl`. Методология и ограничения: `docs/RESEARCH_PROTOCOL_ru.md`.

## Научная честность

`research_only` не должен попадать в LLM при любой ошибке и не должен изображать знания, которых в базе нет. `llm_only` не должен видеть SQLite. Никакие изменения Qwen не считаются научным вкладом индекса. Сравнение делается на одном фиксированном наборе запросов с учётом отказов, времени и точности; ранжировщик v0.3.2 — пока **baseline**, а не доказанная новизна.
