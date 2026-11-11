# Формат задания

Спецификация банка заданий A1 (фаза 2). Её проверяет `fr-drill bank check`, по ней `/exam` собирает тест (фаза 3).
Типы вопросов взяты из [DELF A1](../reference/delf-a1.md), темы — из [карты тем](../syllabus/syllabus.md) (`data/syllabus/topics.json`), составленной по документу «Inventaire linguistique des contenus clés des niveaux du CECRL» (Eaquals, CIEP, Eurocentres, ACPLS, 2015), приложение E.

## Файлы

- `data/bank/<тема>.json` — задания, основная тема которых `<тема>`: `gram.etre.json`, `act.time-price.json`.
- `data/bank/consignes.json` — общие условия: `{"choose-form": {"fr": "Choisissez la bonne forme.", "ru": "Выберите верную форму."}}`.
- Файл темы: `{"docs": [...], "items": [...]}`; `docs` можно опустить.

## Документ (`docs`)

Общий текст для нескольких вопросов: афиша, письмо, объявление, как в DELF.

| Поле | Что |
|---|---|
| `id` | уникален во всём банке |
| `kind` | `courriel`, `affiche`, `annonce`, `carte`, `article`, `message`, `formulaire` |
| `text` | текст документа, переносы строк — `\n` |

## Задание (`items`)

| Поле | Обязательно | Что |
|---|---|---|
| `id` | да | `<слово>-NNN`, уникален во всём банке: `etre-001` |
| `type` | да | `choice`, `short`, `number`, `write` |
| `part` | да | `practice` — грамматика и лексика; `ce` — понимание письменного текста; `pe` — письмо |
| `topics` | да | id из `topics.json`, первая — основная (по ней назван файл) |
| `consigne` | да | `{"fr", "ru"}` или ключ из `consignes.json` |
| `prompt` | да, кроме `write` с `doc` | вопрос или фраза с пропуском `___` |
| `doc` | для `ce` | id документа |
| ответ | да | по типу, см. ниже |
| `explain` | для `practice` | по-русски: почему верно, одно правило |
| `source` | да | `{"by": "author" \| "book" \| "delf-sample" \| "teacher", "ref": "…"}` |

**Ступень задания** — наибольшая ступень его тем. До ступени 2 включительно у условия обязателен `ru`, и тест показывает его под `fr`. С 3-й ступени — только `fr`, как на экзамене.

## Типы и ответы

| `type` | Ответ | Проверка |
|---|---|---|
| `choice` | `options` (2–4, в DELF — 3), `answer` — индекс | Python |
| `short` | `accept` — список допустимых ответов, первый — образцовый | Python, нормализация ниже |
| `number` | `answer`: число (`19`, `12.5`) или время `"19:00"` | Python: `19h`, `19 h`, `19:00`, `12,50 €` |
| `write` | `points` — пункты задания, `min_words` (в DELF — 40) | Python считает слова, оценивает проверяющий по пяти критериям DELF |

**Нормализация `short`:** регистр, лишние пробелы и вид апострофа не важны. Акценты:
- `practice` — пропущенный акцент — ошибка вида «акцент», как в тренажёре спряжения;
- `ce` — засчитывается: проверяется понимание, а не орфография (как в DELF).

**Слова в `write`** — по правилу DELF: слово — то, что разделено пробелами; `j'ai` и `rendez-vous` — одно слово.

## Что проверяет `fr-drill bank check`

Код — `src/fr_drill/bank.py`, тест — `tests/test_bank.py`. Выход 0 — банк годен, иначе список ошибок `файл: id: что не так`.

- id заданий и документов уникальны во всём банке (одно пространство имён); формат `id`;
- нет неизвестных полей — опечатка вроде `explian` не проходит молча;
- темы существуют в `topics.json`; файл назван по первой теме;
- `doc` и ключ `consigne` существуют; у `ce` есть `doc`;
- ступень ≤ 2 — у условия есть `ru`;
- `choice`: варианты различны (и после нормализации), `answer` в пределах;
- `short`: `accept` не пуст, без повторов после нормализации; `number` — число или `ЧЧ:ММ`;
- `practice` — есть `explain`; `choice` и `short` в `practice` — задания на пропуск: в `prompt` ровно один `___`; в остальных — не больше одного;
- `source.by` из списка, `ref` не пуст;
- `write` — есть `points` и `min_words`.

## Пример

Два задания из разных файлов, сведённые в один для краткости: `etre-001` живёт в `gram.etre.json`, `piscine-001` и документ — в `act.time-price.json`.

```json
{
  "docs": [
    {"id": "affiche-piscine", "kind": "affiche",
     "text": "Piscine municipale\nOuverte du lundi au samedi, de 9 h à 19 h.\nEntrée : 3,50 €."}
  ],
  "items": [
    {"id": "etre-001", "type": "choice", "part": "practice", "topics": ["gram.etre", "gram.subject-pronouns"],
     "consigne": "choose-form", "prompt": "Nous ___ étudiants.", "options": ["sommes", "êtes", "sont"], "answer": 0,
     "explain": "nous → sommes: être неправильный — je suis, tu es, il est, nous sommes, vous êtes, ils sont.",
     "source": {"by": "author", "ref": "Inventaire CECRL (Eaquals, CIEP 2015), приложение E: être"}},
    {"id": "piscine-001", "type": "number", "part": "ce", "doc": "affiche-piscine", "topics": ["act.time-price", "lex.time-date"],
     "consigne": {"fr": "Lisez le document et répondez.", "ru": "Прочитайте документ и ответьте."},
     "prompt": "À quelle heure ferme la piscine ?", "answer": "19:00",
     "source": {"by": "author", "ref": "delf-a1: CE ex. 3"}}
  ]
}
```
