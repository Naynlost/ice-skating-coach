"""Komut satiri girisi. `make analyze`, `make overlay` ve `make check` buraya baglanir.

`check` disindaki alt komutlar iskelet asamasinda bilerek uygulanmadi; her biri
kendi gorevinde doldurulur (overlay -> 1.6, analyze -> 3.4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.ingest import IngestConfig, QualityReport, check_quality

_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="core.cli", description="Buz pateni analiz araci")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="tek videodan tam rapor uret")
    analyze.add_argument("--video", required=True, help="girdi mp4 yolu")
    analyze.add_argument("--element", default="jump", help="eleman modulu adi")
    analyze.add_argument("--level", default="beginner", help="sporcu seviyesi")

    overlay = sub.add_parser("overlay", help="poz overlay videosu uret")
    overlay.add_argument("--video", required=True, help="girdi mp4 yolu")
    overlay.add_argument("--frame", type=int, default=None, help="tek kareyi izole et")
    overlay.add_argument("--dump-json", action="store_true", help="ara degerleri JSON dok")

    check = sub.add_parser("check", help="video kalite kontrolu (kabul kapisi)")
    source = check.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="tek video yolu")
    source.add_argument("--dir", help="videolarin bulundugu dizin")
    check.add_argument("--config", default=None, help="ingest.yaml yolu")

    return parser


def _describe(path: Path, report: QualityReport) -> str:
    mark = "OK " if report.accepted else "RET"
    detail = report.summary()
    if report.info is not None:
        info = report.info
        detail = (
            f"{info.width}x{info.height} {info.fps:.0f}fps "
            f"{info.duration_s:.1f}sn | {detail}"
        )
    return f"{mark} {path.name:<28} {detail}"


def _run_check(args: argparse.Namespace) -> int:
    config = IngestConfig.load(Path(args.config)) if args.config else IngestConfig.load()

    if args.video:
        paths = [Path(args.video)]
    else:
        directory = Path(args.dir)
        if not directory.is_dir():
            print(f"dizin yok: {directory}", file=sys.stderr)
            return 2
        paths = sorted(
            p for p in directory.iterdir() if p.suffix.lower() in _VIDEO_SUFFIXES
        )
        if not paths:
            print(f"{directory} icinde video yok", file=sys.stderr)
            return 2

    rejected = 0
    for path in paths:
        report = check_quality(path, config)
        rejected += not report.accepted
        print(_describe(path, report))

    if len(paths) > 1:
        print(f"\n{len(paths)} video, {len(paths) - rejected} kabul, {rejected} ret")
    return 1 if rejected else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        return _run_check(args)
    print(f"[iskelet] '{args.command}' henuz uygulanmadi. Bkz. TASKS.md", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
