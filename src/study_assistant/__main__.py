"""Entry point: `python -m study_assistant`."""

from __future__ import annotations

import sys

from study_assistant.cli import main

if __name__ == "__main__":
    sys.exit(main())
