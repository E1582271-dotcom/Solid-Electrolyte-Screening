"""Command-line interface for frozen and live Project-1 workflows."""
from __future__ import annotations

import argparse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p1screen")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("evaluate", help="run grouped evaluation and strict held-out analysis")
    sub.add_parser(
        "training-curves",
        help="compute the supplementary boosting-curve and learning-curve tables",
    )
    sub.add_parser("refresh-mp", help="refresh the reviewed MP snapshot using MP_API_KEY")
    sub.add_parser("screen", help="fit production ensemble and rebuild ledger/queue")
    figures = sub.add_parser(
        "figures", help="render all figures, one main figure, or only the supplementary figure"
    )
    selection = figures.add_mutually_exclusive_group()
    selection.add_argument("--figure", type=int, choices=range(1, 6))
    selection.add_argument("--supplementary", action="store_true")
    verify = sub.add_parser("verify", help="run the offline release gate")
    verify.add_argument(
        "--frozen", action="store_true",
        help="compare files with the reviewed manifest without modifying it",
    )
    sub.add_parser("manifest", help="explicitly update the reviewed release manifest")
    rebuild = sub.add_parser("rebuild", help="rebuild every derived artifact")
    rebuild.add_argument("--refresh-mp", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        from .evaluation import run_evaluation

        result = run_evaluation()
        print(f"Evaluation complete: strict-test MAE={result['strict_test']['MAE']:.3f}")
    elif args.command == "training-curves":
        from .supplementary import run_training_curves

        summary = run_training_curves()
        final = ", ".join(
            f"{name} {item['final_validation_rmse']:.3f}"
            for name, item in summary["boosting"].items()
        )
        print(f"Training curves complete: {summary['n_fits']} fits; "
              f"final validation RMSE {final}")
    elif args.command == "refresh-mp":
        from .screening import refresh_mp_snapshot

        frame = refresh_mp_snapshot()
        print(f"Frozen MP snapshot: {len(frame)} entries")
    elif args.command == "screen":
        from .screening import build_screen

        ledger, queue = build_screen()
        print(f"Screen complete: {len(ledger)} entries; {len(queue)} candidate compositions")
    elif args.command == "figures":
        from .figures import render_figures, render_supplementary

        if args.supplementary:
            render_supplementary()
        else:
            render_figures(args.figure)
    elif args.command == "verify":
        from .verify import verify_release

        verify_release(frozen=args.frozen)
    elif args.command == "manifest":
        from .verify import verify_release

        verify_release(write_manifest=True)
    elif args.command == "rebuild":
        from .config import SOURCE_DATA

        SOURCE_DATA.mkdir(parents=True, exist_ok=True)
        for source in SOURCE_DATA.glob("*.csv"):
            source.unlink()
        if args.refresh_mp:
            from .screening import refresh_mp_snapshot

            refresh_mp_snapshot()
        from .evaluation import run_evaluation
        from .figures import render_figures
        from .screening import build_screen
        from .supplementary import run_training_curves
        from .verify import verify_release

        run_evaluation()
        run_training_curves()
        build_screen()
        render_figures()
        verify_release()


if __name__ == "__main__":
    main()
