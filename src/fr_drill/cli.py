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
    return parser


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


if __name__ == "__main__":
    main()
