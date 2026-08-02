"""Backward-compatible entry point for the generic LeRobot RICL corpus builder."""

try:
    from .build_lerobot_ricl_corpus import build_corpus
    from .build_lerobot_ricl_corpus import main
except ImportError:
    from build_lerobot_ricl_corpus import build_corpus
    from build_lerobot_ricl_corpus import main


if __name__ == "__main__":
    main()
