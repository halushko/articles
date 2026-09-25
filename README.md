# Fragment Segmenter MVP

MVP для сегментації plain text / markdown документації програмного продукту на логічні `Fragment`.

## Що робить

- обробляє plain text / markdown;
- виділяє headings, paragraphs, list items;
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
