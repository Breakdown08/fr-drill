# fr-drill

A trainer for French at level A1: a browser drill with ten tabs and a test modelled on the DELF A1 exam.
The interface and the explanations are in Russian, because the trainer is written for Russian speakers;
the code, this README and the command line are in English.

- `drill`: ten tabs, each a different skill: conjugation of regular `-er` verbs, A1 words (pick the
  translation, then the article), numbers 0..9999 built by rule, negation, irregular verbs
  (être, avoir, aller, faire, prendre, venir), building a sentence from tiles, à and de, pronouns,
  the gender of nouns, articles. Every answer is checked by Python and remembered; what went wrong
  comes back more often.
- `exam`: a test built from a bank of tasks, in the order and the types of DELF A1: grammar and
  vocabulary practice, reading comprehension, a short letter.
- `bank`, `syllabus`: validation of the task bank and the map of A1 topics it is built on.
- `overlap`: a check that data files do not repeat a textbook word for word (see Sources).

Not published yet: a vocabulary file (see Vocabulary). Without it `drill` does not start.

## Run

Python 3.14 and [uv](https://docs.astral.sh/uv/) are needed. Run from a clone of the repository: the
data lives in `data/` next to the code, and the program looks for it there.

```
uv sync
uv run fr-drill drill              # the drill, opens the browser on http://127.0.0.1:8770
uv run fr-drill exam --stage 1     # a test of the topics of stage 1, http://127.0.0.1:8771
```

Other commands:

```
uv run fr-drill drill --check      # validate the data files
uv run fr-drill drill --stats      # progress on the command line
uv run fr-drill exam last          # score of the last test
uv run fr-drill bank check         # validate the task bank
uv run fr-drill syllabus [--out PATH]
uv run fr-drill overlap BOOK FILE [FILE ...] [--run 8] [--sentence 5]
```

Everything listens on `127.0.0.1` only. Add `--no-open` to keep the browser closed and `--port` to
change the port.

## Vocabulary

The words tab, the gender tab and the sentence tab need a vocabulary file. **There is no open
vocabulary in this repository yet, and without the file `fr-drill drill` stops with a message.**
Put your own file at `data/drills/vocab.json` (it is in `.gitignore`) or point `FR_DRILL_VOCAB` to
it. Format:

```json
{
  "lessons": [
    {"n": 1, "ru": "Приветствия", "words": [
      {"fr": "bonjour", "ru": "здравствуйте"},
      {"fr": "le café", "ru": "кофе", "g": "m"},
      {"fr": "l'Espagne", "ru": "Испания", "g": "f", "definite": true}
    ]},
    {"n": 4, "ru": "Числа", "own_drill": "numbers", "words": []}
  ]
}
```

- `n` is the number of the lesson, `ru` its title (the topic of its words).
- A word has `fr` and `ru`; `g` is `m`, `f` or `mf`; a noun starting with `l'` or `les` needs `g`;
  `definite` marks a name that always takes its article (l'Espagne).
- A lesson with `own_drill` belongs to another tab (numbers) and gives no word cards.
- Each word is one card and appears once in the file.

## Personal data

Progress is written as JSON lines to `progress/` (one file per tab, `exam.jsonl` and
`exam-grades.jsonl` for the test). Set `FR_DRILL_PROGRESS` to keep it elsewhere. The directory is
created on the first answer.

`books/` and `index/` are reserved for a helper over textbooks that does not exist yet. All of
`progress/`, `books/`, `index/` and `data/drills/vocab.json` are in `.gitignore`.

Both variables are read once, when the package is imported. A relative path counts from the current
directory.

## Grading a letter

The test has one task that a program cannot grade: a letter of at least 40 words. The test counts
the words and keeps the letter. A reviewer (a teacher, or any assistant) grades it:

1. `uv run fr-drill exam pending` prints the letters waiting for a grade as JSON, with the task, the
   words and the five DELF criteria.
2. The reviewer returns a grade as JSON: scores 0..3 for each criterion, a comment, fixes.
3. `uv run fr-drill exam grade FILE` (or `grade -` from stdin) writes it down; `exam last` shows it.

## What it does not do

- There is no speaking part. DELF A1 has one; a teacher has to do it.
- Listening is trained only with numbers, and it uses the voice of the browser, so it works only if
  the browser has a French voice.
- A letter is not graded by the program (see above).
- The explanations are in Russian only.
- There is no assistant over textbooks and no other language backend yet.

## Sources and licenses

Code and own data are under Apache 2.0 (see `LICENSE`).

| Data | Source | Status |
|---|---|---|
| `verbs-er`, `negation-er`, `etre-avoir`, `sentences`, `a-de`, `pronouns`, `numbers`, the task bank | own | Apache 2.0 |
| `data/syllabus/topics.json` | own text; the map of topics follows *Inventaire linguistique des contenus clés des niveaux du CECRL* (Eaquals, CIEP, Eurocentres, ACPLS, 2015), appendix E | Apache 2.0 |
| `data/reference/delf-a1.md` | own summary of the format of DELF A1 | Apache 2.0 |
| `a-de-verbs.json` | verbs and their government checked against [irgol.ru](https://irgol.ru/grammaticheskij-spravochnik/upravlenie-glagolov-i-glagolnyh-vyr/); the sentences are own | Apache 2.0 |
| `articles.json`, `gender.json` | explanations after Annie Heminway, *Complete French All-in-One* (2018): chapter references only, no text reproduced; checked with `fr-drill overlap` | Apache 2.0 |

`fr-drill overlap` compares the strings of JSON files with the text of a book (PDF or text) and
reports every sentence of five words or more found in the book whole and every run of eight words
that they share. It reads the book from a path you give; the book itself never enters the
repository.

## Tests

```
uv run ruff check && uv run ruff format --check && uv run ty check && uv run mypy && uv run pytest
```

Tests that need a vocabulary file are skipped when there is none.

## License

Apache License 2.0. See `LICENSE`.
