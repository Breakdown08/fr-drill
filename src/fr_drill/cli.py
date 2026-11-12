"""The fr-drill command line."""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="fr-drill", description="French A1 drill and DELF-style exam trainer"
    )
    sub: argparse._SubParsersAction[argparse.ArgumentParser] = parser.add_subparsers(
        dest="command", required=True
    )
    drill: argparse.ArgumentParser = sub.add_parser(
        "drill",
        help="French drill in the browser: présent -er, A1 words, numbers, negation, sentences, à and de",
    )
    drill.add_argument("--port", type=int, default=8770, help="port on 127.0.0.1 (default 8770)")
    drill.add_argument(
        "--check",
        action="store_true",
        help="only validate the verb list, word cards, number rule, negation phrases",
    )
    drill.add_argument("--stats", action="store_true", help="accuracy so far, by person and mistake")
    drill.add_argument("--no-open", action="store_true", help="do not open the browser")
    bank: argparse.ArgumentParser = sub.add_parser("bank", help="French task bank")
    bank.add_argument("action", choices=["check"], help="check: validate data/bank/*.json against format.md")
    exam: argparse.ArgumentParser = sub.add_parser(
        "exam", help="test from the French task bank in the browser"
    )
    exam.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=["run", "last", "pending", "grade"],
        help="run (default): serve a test; last: the last test; "
        "pending: letters waiting for a grade (JSON); grade: a grade of a letter (JSON file or -)",
    )
    exam.add_argument("arg", nargs="?", help="grade: reply file or - (stdin)")
    exam.add_argument("--stage", type=int, default=1, help="items up to this stage (default 1)")
    exam.add_argument("--port", type=int, default=8771, help="port on 127.0.0.1 (default 8771)")
    exam.add_argument("--no-open", action="store_true", help="do not open the browser")
    syl: argparse.ArgumentParser = sub.add_parser(
        "syllabus", help="check topics.json and generate syllabus.md"
    )
    syl.add_argument("--check", action="store_true", help="only validate topics.json")
    syl.add_argument("--out", type=Path, help="write the note here instead of data/syllabus/syllabus.md")
    ovl: argparse.ArgumentParser = sub.add_parser(
        "overlap", help="find verbatim overlaps between JSON data files and a textbook"
    )
    ovl.add_argument("book", type=Path, help="the textbook: a PDF or a text file")
    ovl.add_argument("files", type=Path, nargs="+", metavar="FILE", help="JSON data files to check")
    ovl.add_argument("--run", type=int, default=8, help="shortest common run of words (default 8)")
    ovl.add_argument(
        "--sentence", type=int, default=5, help="shortest French sentence found whole (default 5)"
    )
    idx: argparse.ArgumentParser = sub.add_parser("index", help="index a textbook PDF for search")
    idx.add_argument("pdf", type=Path, help="the textbook PDF")
    sel: argparse._MutuallyExclusiveGroup = idx.add_mutually_exclusive_group()
    sel.add_argument("--pages", help="printed page labels to index, A-B or A")
    sel.add_argument("--chapter", help="top-level chapter number to index")
    _add_store(idx)
    sea: argparse.ArgumentParser = sub.add_parser("search", help="find textbook passages for a query")
    sea.add_argument("query", help="the query, in any language")
    sea.add_argument("--k", type=int, default=5, help="number of passages (default 5)")
    sea.add_argument("--json", action="store_true", help="print JSON")
    _add_store(sea)
    ask: argparse.ArgumentParser = sub.add_parser(
        "ask", help="answer a question from the textbook, with page references"
    )
    ask.add_argument("question", help="the question")
    ask.add_argument("--backend", choices=["ollama", "claude"], default="ollama", help="default ollama")
    ask.add_argument("--model", help="model name for the backend")
    ask.add_argument("--k", type=int, default=5, help="number of passages (default 5)")
    _add_store(ask)
    ev: argparse.ArgumentParser = sub.add_parser("eval", help="recall@k of search over a question set")
    ev.add_argument("file", nargs="?", type=Path, help="eval JSON (default data/rag/eval.json)")
    ev.add_argument("--k", type=int, default=5, help="top k (default 5)")
    ev.add_argument("--json", action="store_true", help="print JSON")
    _add_store(ev)
    return parser


def _add_store(p: argparse.ArgumentParser) -> None:
    p.add_argument("--index", type=Path, help="index directory (default index/)")
    p.add_argument("--qdrant-url", help="use a Qdrant server instead of the local index")


def main(argv: list[str] | None = None) -> None:
    args: argparse.Namespace = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "drill":
        from fr_drill import drill

        raise SystemExit(drill.main(args.port, args.check, args.stats, not args.no_open))
    elif args.command == "bank":
        from fr_drill import bank

        raise SystemExit(bank.main())
    elif args.command == "exam":
        from fr_drill import exam

        raise SystemExit(exam.main(args.action, args.arg, args.stage, args.port, not args.no_open))
    elif args.command == "syllabus":
        from fr_drill import syllabus

        raise SystemExit(syllabus.main(args.check, out=args.out))
    elif args.command == "overlap":
        from fr_drill import overlap

        raise SystemExit(overlap.main(args.book, args.files, args.run, args.sentence))
    elif args.command in ("index", "search", "ask", "eval"):
        try:
            from fr_drill import rag
        except ImportError:
            print("fr-drill: install the helper with: uv sync --extra rag", file=sys.stderr)
            raise SystemExit(2) from None
        index: Path = args.index if args.index is not None else rag.INDEX
        if args.command == "index":
            raise SystemExit(rag.cmd_index(args.pdf, args.pages, args.chapter, index, args.qdrant_url))
        elif args.command == "search":
            raise SystemExit(rag.cmd_search(args.query, args.k, args.json, index, args.qdrant_url))
        elif args.command == "ask":
            raise SystemExit(
                rag.cmd_ask(args.question, args.backend, args.model, args.k, index, args.qdrant_url)
            )
        else:
            file: Path = args.file if args.file is not None else rag.EVAL
            raise SystemExit(rag.cmd_eval(file, args.k, args.json, index, args.qdrant_url))


if __name__ == "__main__":
    main()
