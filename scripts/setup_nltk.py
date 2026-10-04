"""
Download the NLTK data used by the legacy Stage 1 module (src/data/preprocess.py) and its tests.

The Phase 1 URL pipeline does not use NLTK. Data is stored inside the active virtualenv
(<sys.prefix>/nltk_data), which NLTK searches automatically, so nothing is written to $HOME.

Usage:  .venv/bin/python scripts/setup_nltk.py
"""

import os
import sys

import nltk

RESOURCES = ["punkt_tab", "stopwords", "wordnet"]


def main() -> int:
    target = os.path.join(sys.prefix, "nltk_data")
    os.makedirs(target, exist_ok=True)
    ok = True
    for resource in RESOURCES:
        ok &= bool(nltk.download(resource, download_dir=target, quiet=True))
        print(f"{resource}: {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
