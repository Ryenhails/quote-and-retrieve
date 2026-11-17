"""Path defaults shared by every script. Import this module before using any path.

Three environment variables locate everything; each has a default, so a plain clone works:

  PROJECT_ROOT       the repository root. data/, outputs/, and logs/ live under it.
  QR_CACHE           model snapshots and block-embedding caches. Default: <PROJECT_ROOT>/../cache
  QWEN3VL_EMB_REPO   a directory that contains scripts/qwen3_vl_embedding.py, the wrapper class
                     shipped inside the Qwen/Qwen3-VL-Embedding-2B model repository.
                     Default: <QR_CACHE>/Qwen3-VL-Embedding-2B  (see docs/reproduction_details.md)
"""
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PROJECT_ROOT", _REPO)
os.environ.setdefault("QR_CACHE", os.path.join(os.environ["PROJECT_ROOT"], "..", "cache"))
os.environ.setdefault("QWEN3VL_EMB_REPO", os.path.join(os.environ["QR_CACHE"], "Qwen3-VL-Embedding-2B"))

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
QR_CACHE = os.environ["QR_CACHE"]
QWEN3VL_EMB_REPO = os.environ["QWEN3VL_EMB_REPO"]
