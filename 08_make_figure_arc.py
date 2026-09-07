"""Compatibility wrapper for the five-figure renderer."""
import sys

from p1screen.cli import main

if __name__ == "__main__":
    main(["figures", *sys.argv[1:]])
