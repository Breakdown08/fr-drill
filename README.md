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

## Textbook helper

Ask a question about a textbook (in Russian, for example) and get an explanation with page references.
The helper reads a PDF, splits it into chunks that never cross a page, embeds them, stores them in a
Qdrant index and, for `ask`, passes the best chunks to a language model.

```
uv sync --extra rag
uv run --extra rag fr-drill index books/heminway.pdf --chapter 21   # or --pages 1-12 (printed page labels)
uv run --extra rag fr-drill search "c'est или il est перед прилагательным" --k 3
uv run --extra rag fr-drill ask "почему c'est, а не il est?" --backend ollama
uv run --extra rag fr-drill eval data/rag/eval.json --k 5
```

- Embeddings: `intfloat/multilingual-e5-large` (ONNX build `Qdrant/multilingual-e5-large-onnx`, run by
  fastembed, no torch), about 2.2 GB, downloaded once to `index/models/`. Checked with revision
  `ac6781cd1cf88b8306a536d7c9d18a5bd57cc14b` of `Qdrant/multilingual-e5-large-onnx`; SHA-256 of
  `model.onnx` `1c09780c907c8a91a77a6ab1fd231f79e090d2907ca431223703dfebeed3d36c`, of
  `model.onnx_data` `0cf1883fee81c63819a44e2ba0efa51d4043d9759685a4ebebbde97e0623d15c`.
- The index is a local Qdrant in `index/qdrant/`. To use a server instead, add `--qdrant-url URL` to
  every command.
- Backend `ollama` (default model `gemma3:4b`, `--model` changes it) talks to `http://localhost:11434`:

  ```
  docker run -d -p 11434:11434 -v ollama:/root/.ollama --name ollama ollama/ollama
  docker exec ollama ollama pull gemma3:4b
  ```

- Ollama needs no Python package: the helper only calls its HTTP API.
- Backend `claude` runs the Claude Code CLI (`claude -p`), which has to be installed.
- The eval file `data/rag/eval.json` is `{"items": [{"q": "...", "pages": ["308"]}, ...]}` with printed
  page labels. `fr-drill eval` prints recall@k and the share of questions with at least one hit.
  On chapter 21 of the Heminway textbook (30 pages, 62 chunks) the eval set in the repository gives
  recall@5 0.955 and hit@5 1.000 over 11 questions: `fr-drill index <pdf> --chapter 21`, then the
  command above. One chapter is a small haystack; the figure says the pipeline works, not that it
  scales to the whole book.

### Evaluating the answers

`fr-drill eval` measures only search. `fr-drill answer-eval` runs a question set through the same
pipeline as `ask` (search, prompt, generator), grades every answer with code and saves the run:

```
uv run --extra rag fr-drill answer-eval --backend ollama --out runs/gemma-q4.json
uv run --extra rag fr-drill answer-eval --backend claude --model haiku --limit 3
uv run fr-drill answer-report runs/gemma-q8.json runs/gemma-q4.json
```

- `data/rag/answers.json` has 74 questions about chapter 21 with checkable facts and 5 questions outside
  the textbook, for which the right answer is the refusal. An item is
  `{"id", "q", "chapter", "pages", "must", "forbid"}`. `must` is a list of facts, a fact is a list of
  alternatives, and a fact is covered when the answer contains at least one of them. A pattern is a
  string of at most 4 words; a final `*` means "beginning of a word" (`женск*` matches `женского`).
  Matching ignores case, `ё`/`е`, quotes and markdown emphasis. `forbid` lists wrong forms: finding one
  is an error. An item with `"expect": "no_answer"` is right only if the answer is the refusal.
  `fr-drill eval data/rag/answers.json` works too: it skips the `no_answer` items.
- Runs go to `runs/` (in `.gitignore`: they hold quotes from the textbook). A run keeps answers, grades,
  search and generation times. `answer-report A [B]` re-grades saved answers against the current
  question set, no generation needed; for two runs it prints the paired difference of the scores with a
  95% bootstrap interval. It needs no extra.
- Metrics: `score` is the mean share of covered facts (the main one); `pass` the share of answers with
  all facts and no wrong form; `wrong` the share with a wrong form; `false-refusal` the share of answers
  that refuse although the textbook has the answer; `no-answer accuracy` the share of out-of-book
  questions refused; `cited` the share of answers with a page reference `(с. N)`; `in-context` the share
  of cited pages that were among the retrieved ones; `expected-page` the share of answers citing a page
  where the answer is.
- Generation is greedy for Ollama (temperature 0, seed 0). `claude -p` has no temperature flag, so its
  runs are not repeatable.
- Limits: string matching is not understanding. A correct explanation in other words can lose points, a
  wrong one that contains the right forms can earn them. Diacritics are strict (`a pied` is not
  `à pied`). With 74 questions only a difference of about 0.07 to 0.10 in the mean score can be told
  from noise.

### Measured

Chapter 21 of Heminway (30 pages, 62 chunks), k=5, 74 answer items and 5 out-of-book items, on a
laptop CPU (Intel Core Ultra 7 155H, 22 threads, no GPU), 2026-11-18. Search alone on the same 74
questions: recall@5 0.912, hit@5 0.946 (`fr-drill eval data/rag/answers.json`).

| | `gemma3:4b` (Ollama, CPU) | `haiku` (`claude -p`) |
|---|---|---|
| score | 0.359 | 0.881 |
| pass | 0.284 | 0.851 |
| wrong | 0.000 | 0.000 |
| false-refusal | 0.365 | 0.108 |
| no-answer accuracy (5) | 1.000 | 1.000 |
| cited | 0.662 | 0.905 |
| in-context | 0.969 | 1.000 |
| expected-page | 0.568 | 0.905 |
| generation, median per answer | 22.1 s | 6.4 s |

- Haiku minus gemma: score +0.521, 95% interval [+0.417, +0.626], better on 45 questions, worse on
  none (`fr-drill answer-report runs/gemma.json runs/haiku.json`).
- Two full gemma runs gave identical answers on all 79 items (difference 0.000): greedy Ollama runs
  can be compared with each other.
- Gemma mostly loses by refusing (27 of 74) or by giving the bare rule without the example forms
  the item asks for; it never gave a forbidden form. Of the 8 haiku refusals, 4 are search misses
  (the page with the answer was not in the top 5); in the other 4 the right page was retrieved, but
  not necessarily the chunk with the answer.
- After reading the answers by eye, items that asked only for a preposition or a meaning were relaxed
  to accept the bare answer (`предлог sur`), and one item that gave points to an off-topic answer was
  tightened; the figures above are re-graded with the current set.
- The local Qdrant index can be open in one process only. To run two evals at once, copy `index/`
  and pass `--index` to one of them, or use `--qdrant-url`.
- `fr-drill overlap` reports two sentences in the facts, `c'est interdit de fumer` and
  `c'est dur de plaire`: the apostrophe splits `c'est` into two words, so a four-word fact counts as
  a five-word sentence. Both are kept as short example forms.

## Personal data

Progress is written as JSON lines to `progress/` (one file per tab, `exam.jsonl` and
`exam-grades.jsonl` for the test). Set `FR_DRILL_PROGRESS` to keep it elsewhere. The directory is
created on the first answer.

`books/` holds your textbook PDFs and `index/` the search index and the embedding model of the textbook
helper (see below); `runs/` holds the saved runs of `fr-drill answer-eval`. All of
`progress/`, `books/`, `index/`, `runs/` and `data/drills/vocab.json` are in `.gitignore`.

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
- The textbook helper has no "I don't know" threshold: it says there is no answer only when the index is
  empty, otherwise the prompt asks the model to be honest.
- The answer eval checks forms and concepts by string matching, not the quality of an explanation.
- Running headers and footers of the PDF end up in the chunks.
- There is no OCR: a scanned PDF without a text layer gives nothing to index.

## Sources and licenses

Code and own data are under Apache 2.0 (see `LICENSE`).

| Data | Source | Status |
|---|---|---|
| `verbs-er`, `negation-er`, `etre-avoir`, `sentences`, `a-de`, `pronouns`, `numbers`, the task bank | own | Apache 2.0 |
| `data/syllabus/topics.json` | own text; the map of topics follows *Inventaire linguistique des contenus clés des niveaux du CECRL* (Eaquals, CIEP, Eurocentres, ACPLS, 2015), appendix E | Apache 2.0 |
| `data/reference/delf-a1.md` | own summary of the format of DELF A1 | Apache 2.0 |
| `a-de-verbs.json` | verbs and their government checked against [irgol.ru](https://irgol.ru/grammaticheskij-spravochnik/upravlenie-glagolov-i-glagolnyh-vyr/); the sentences are own | Apache 2.0 |
| `data/rag/answers.json` | own questions; the facts are short forms (up to 4 words) from chapter 21 of Heminway, *Complete French All-in-One* (2018); checked with `fr-drill overlap` | Apache 2.0 |
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
