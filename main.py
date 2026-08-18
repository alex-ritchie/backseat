"""Entry point. Run:  uv run main.py  (add --debug for the raw send/reply log)"""
import argparse
import sys

from PyQt6.QtWidgets import QApplication

from gui import ControlPanel


def main() -> int:
    parser = argparse.ArgumentParser(description="Backseat")
    parser.add_argument(
        "--debug", action="store_true",
        help="show a third column logging every send and raw model reply",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    panel = ControlPanel(debug=args.debug)
    panel.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
