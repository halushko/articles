# Documentation Fragmentation MVP

MVP для сегментації plain text / markdown документації програмного продукту на логічні `Fragment`.

## Web MVP у Docker Compose

Web-застосунок дає змогу запустити сегментацію без командного рядка:

- обрати вбудований англомовний набір D1–D5;
- або завантажити ZIP з UTF-8 Markdown-документами;
- зберегти результати для кожної версії документа в PostgreSQL `JSONB`;
- завантажити результат як ZIP з окремими JSON-файлами та маніфестом.

### Запуск

```bash
cp .env.example .env
```

Задайте у `.env` локальний пароль PostgreSQL, після чого виконайте:

```bash
docker compose up --build
```

UI буде доступний за адресою <http://localhost:8080>. Під час старту застосунок
автоматично виконує Alembic-міграції. Дані PostgreSQL зберігаються в named
volume `postgres-data`.

Зупинити сервіси:

```bash
docker compose down
```

Видалити також локальні дані PostgreSQL:

```bash
docker compose down --volumes
```

### API

Створення запуску на вбудованих документах:

```bash
curl -X POST http://localhost:8080/api/v1/runs \
  -F source=example
```

Завантаження власної документації:

```bash
curl -X POST http://localhost:8080/api/v1/runs \
  -F source=upload \
  -F archive=@documentation.zip
```

Додаткові endpoints:

```text
GET /api/v1/runs/{run_id}
GET /api/v1/runs/{run_id}/result
GET /health/live
GET /health/ready
GET /docs
```

### Версіонування і кеш

SHA-256 обчислюється з початкових байтів кожного документа. Але ключ кешу
сегментації містить три компоненти:

```text
document_sha256 + segmenter_version + config_sha256
```

Тому зміна алгоритму або його конфігурації створить новий результат навіть для
того самого документа. Хеш усього корпусу обчислюється з відсортованого списку
`path + document_sha256`, а не з байтів ZIP: службові timestamps або порядок
файлів в архіві не змінюють ідентичність набору.

Оригінальні завантажені документи не зберігаються. PostgreSQL містить їхні
хеші, метадані запуску і JSON-картки. ZIP результату формується з БД на запит.

### Структура результату

```text
result.zip
├── manifest.json
├── documents/
│   ├── 001_document.<hash>.fragments.json
│   └── ...
├── all_fragments.json
└── errors.json
```

Поточні обмеження upload-пайплайна: лише `.md`, UTF-8, до 100 документів,
20 MB для ZIP і 50 MB після розпакування. Перевіряються небезпечні шляхи,
символічні посилання, шифровані записи, дублікати та unsupported-формати.

## Що робить

- обробляє plain text / markdown;
- виділяє headings, paragraphs, list items і окремі рядки Markdown-таблиць;
- розпізнає user stories;
- розпізнає acceptance criteria;
- розпізнає API operations;
- виконує sentence segmentation з винятками для `e.g.`, `i.e.`, `etc.`, API paths, versions;
- виконує `hard_process_split` для `and then`, `then`, `after that`, `next`;
- виконує `soft_process_split` для `if`, `when`, `unless`;
- повертає список `Fragment`.

## Встановлення

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Для тестів:

```bash
pip install -e ".[dev]"
pytest
```

## Запуск прикладу

```bash
python -m fragment_segmenter.cli examples/checkout.md --artifact-id checkout_doc --format markdown
```

Або після встановлення entrypoint:

```bash
fragment-segment examples/checkout.md --artifact-id checkout_doc --format markdown
```

## Optional Unstructured adapter

У MVP основний шлях — rule-based parser, бо він краще зберігає line/char source positions.

Адаптер `UnstructuredPartitioner` залишено як optional extension:

```bash
pip install -e ".[unstructured]"
```

Він може бути корисний для майбутнього DOCX/PDF/HTML, але кінцеві `Fragment` все одно формуються внутрішнім шаром методу.

## Реалістичний корпус документації

Нерозмічений англомовний корпус D1–D5 розміщено в
`examples/access_recovery_source_docs/`. У цих файлах немає ідентифікаторів
вершин і ребер, повної таблиці переходів або готових кандидатів на агрегацію.
Цей каталог є вбудованою тестовою документацією Web MVP і призначений для
подальшого автоматичного виділення операцій та відновлення зв'язків між ними.

Наприклад:

```bash
fragment-segment \
  examples/access_recovery_source_docs/D1_access_recovery_support_policy.md \
  --pretty
```
