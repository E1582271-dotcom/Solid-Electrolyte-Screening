"""Compatibility wrapper for the offline release gate."""
from p1screen.cli import main

if __name__ == "__main__":
    main(["verify", "--frozen"])
