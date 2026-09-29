#!/usr/bin/env python3
"""Download the MiniLM ONNX embedder used by the API (no PyTorch).

Render's build runs this so the first chat does not have to fetch weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from minilm_onnx import ensure_minilm_files


def main() -> None:
    dest = ensure_minilm_files()
    print("MiniLM ONNX ready:", dest)


if __name__ == "__main__":
    main()
