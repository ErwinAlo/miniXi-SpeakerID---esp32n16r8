from __future__ import annotations

import ast
import struct
from pathlib import Path


try:
    Import("env")  # type: ignore[name-defined]
except Exception:
    env = None

if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
elif env is not None:
    SCRIPT_DIR = Path(env.subst("$PROJECT_DIR")) / "Train"
else:
    SCRIPT_DIR = Path.cwd() / "Train"

REPO_ROOT = SCRIPT_DIR.parent
LABELS_NPY = SCRIPT_DIR / "labels.npy"
PROCESS_LOG = SCRIPT_DIR / "procesamiento_log.txt"
HEADER_PATH = REPO_ROOT / "src" / "speaker_labels.h"


def load_labels_with_numpy(path: Path) -> list[str] | None:
    try:
        import numpy as np
    except ImportError:
        return None

    labels = np.load(path, allow_pickle=True)
    return [str(label) for label in labels.tolist()]


def load_unicode_npy(path: Path) -> list[str] | None:
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            return None

        major, minor = struct.unpack("BB", f.read(2))
        if (major, minor) == (1, 0):
            header_len = struct.unpack("<H", f.read(2))[0]
        elif major in (2, 3):
            header_len = struct.unpack("<I", f.read(4))[0]
        else:
            return None

        header = ast.literal_eval(f.read(header_len).decode("latin1"))
        descr = header.get("descr", "")
        shape = header.get("shape", ())
        count = shape[0] if len(shape) == 1 else 0

        if not isinstance(count, int) or count <= 0:
            return None

        if descr.startswith("<U"):
            chars_per_item = int(descr[2:])
            item_size = chars_per_item * 4
            return [
                f.read(item_size).decode("utf-32le").rstrip("\x00")
                for _ in range(count)
            ]

        if descr.startswith(">U"):
            chars_per_item = int(descr[2:])
            item_size = chars_per_item * 4
            return [
                f.read(item_size).decode("utf-32be").rstrip("\x00")
                for _ in range(count)
            ]

        if descr.startswith("|S"):
            item_size = int(descr[2:])
            return [
                f.read(item_size).rstrip(b"\x00").decode("utf-8")
                for _ in range(count)
            ]

    return None


def load_labels_from_log(path: Path) -> list[str] | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Clases encontradas:") or line.startswith("Clases:"):
            _, raw_labels = line.split(":", 1)
            try:
                labels = ast.literal_eval(raw_labels.strip())
            except (SyntaxError, ValueError):
                continue

            if isinstance(labels, list) and all(isinstance(label, str) for label in labels):
                return labels

    return None


def load_labels() -> list[str]:
    if LABELS_NPY.exists():
        labels = load_labels_with_numpy(LABELS_NPY)
        if labels:
            return labels

        labels = load_unicode_npy(LABELS_NPY)
        if labels:
            return labels

    if PROCESS_LOG.exists():
        labels = load_labels_from_log(PROCESS_LOG)
        if labels:
            return labels

    raise RuntimeError(
        "No se pudieron obtener etiquetas desde Train/labels.npy ni Train/procesamiento_log.txt"
    )


def c_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_header(labels: list[str]) -> str:
    labels_block = ",\n".join(f"    {c_string(label)}" for label in labels)
    return f"""#pragma once

// Auto-generated from Train/labels.npy by Train/generate_speaker_labels.py.
// Keep this file synchronized with the trained model.

constexpr int NUM_CLASSES = {len(labels)};

static const char* const SPEAKER_LABELS[NUM_CLASSES] = {{
{labels_block}
}};
"""


def main() -> None:
    labels = load_labels()
    HEADER_PATH.write_text(build_header(labels), encoding="utf-8", newline="\n")
    print(f"Generated {HEADER_PATH.relative_to(REPO_ROOT)} with labels: {labels}")


main()
