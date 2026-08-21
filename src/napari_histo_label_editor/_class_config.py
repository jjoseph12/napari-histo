"""Read and write label-class configuration files.

The helpers in this module deliberately avoid Qt and napari imports so class
configuration can be validated and saved without constructing the editor UI.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile
from typing import Mapping, Union

from matplotlib.colors import to_hex, to_rgba

from ._io import FileIdentity, file_identity


PathLike = Union[str, os.PathLike[str]]

__all__ = [
    "atomic_write_class_config",
    "normalize_color",
    "read_class_config",
]


def normalize_color(color: object) -> str | None:
    """Return *color* as lowercase ``#rrggbb``, or ``None`` if invalid.

    Matplotlib color names, short/long hexadecimal values, and its other
    supported color specifications are accepted.  Alpha, when supplied, is
    intentionally discarded because the label layer controls opacity.
    """
    if color is None:
        return None

    if isinstance(color, str):
        color = color.strip()
        if not color:
            return None

    try:
        return to_hex(to_rgba(color), keep_alpha=False).lower()
    except (TypeError, ValueError):
        return None


def read_class_config(path: PathLike) -> tuple[dict[int, str], dict[int, str]]:
    """Read class names and optional colors from a CSV file.

    For compatibility with existing mapping files, the first two columns are
    always interpreted as the integer label value and class name, regardless
    of their headings.  A column headed ``color`` (case-insensitively) may
    appear anywhere after them.  Missing or invalid colors are omitted so the
    caller can use its normal palette fallback.

    Returns
    -------
    class_map, class_colors
        Dictionaries keyed by integer label value.  Label zero is added as
        ``"background"`` when it is absent from the CSV.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("Class mapping CSV is empty") from error

        if len(header) < 2:
            raise ValueError("Class mapping CSV must contain at least two columns")

        color_index = next(
            (
                index
                for index, heading in enumerate(header)
                if index >= 2 and heading.strip().casefold() == "color"
            ),
            None,
        )

        class_map: dict[int, str] = {}
        class_colors: dict[int, str] = {}

        for line_number, row in enumerate(reader, start=2):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"Class mapping CSV row {line_number} must contain at least "
                    "two columns"
                )

            raw_value = row[0].strip()
            try:
                value = int(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"Invalid integer label value {raw_value!r} on row "
                    f"{line_number}"
                ) from error

            class_map[value] = row[1]

            # A later duplicate row represents the complete, authoritative
            # record for that value, including the absence of a valid color.
            class_colors.pop(value, None)
            if color_index is not None and color_index < len(row):
                normalized = normalize_color(row[color_index])
                if normalized is not None:
                    class_colors[value] = normalized

    class_map.setdefault(0, "background")
    return class_map, class_colors


def atomic_write_class_config(
    path: PathLike,
    class_map: Mapping[int, str],
    class_colors: Mapping[int, object] | None = None,
    *,
    expected_identity: FileIdentity | None = None,
) -> Path:
    """Atomically write a standardized class-configuration CSV.

    The completed temporary file is flushed and replaced from the same
    directory, so readers never observe a partially written configuration.
    Colors that are absent or invalid are written as empty cells and therefore
    use the editor's palette fallback when loaded again.
    When ``expected_identity`` is supplied, the destination must remain the
    exact file that was loaded both before writing the temporary CSV and
    immediately before its atomic replacement.
    """
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise ValueError(
            "Class configuration destination must be an absolute path"
        )

    # Resolve before creating the same-directory temporary file.  In
    # particular, this preserves an existing symlink and atomically replaces
    # its referent instead of replacing the symlink itself.
    target = target.resolve(strict=False)
    if expected_identity is not None:
        _verify_expected_identity(target, expected_identity)

    classes = {int(value): str(name) for value, name in class_map.items()}
    classes.setdefault(0, "background")
    colors = (
        {}
        if class_colors is None
        else {int(value): color for value, color in class_colors.items()}
    )

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("value", "class_name", "color"))
            for value in sorted(classes):
                color = normalize_color(colors.get(value)) or ""
                writer.writerow((value, classes[value], color))
            stream.flush()
            os.fsync(stream.fileno())

        if expected_identity is not None:
            _verify_expected_identity(target, expected_identity)
        os.replace(temporary_path, target)
    except BaseException:
        # os.fdopen takes ownership of the descriptor, except if its own
        # construction fails.  Closing an already closed descriptor is safe to
        # attempt here and prevents a leak on that rare path.
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise

    return target


def _verify_expected_identity(
    target: Path,
    expected_identity: FileIdentity,
) -> None:
    """Refuse to overwrite a mapping file replaced since it was loaded."""
    try:
        current_identity = file_identity(target)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Class configuration destination no longer exists: {target}"
        ) from error

    if current_identity != expected_identity:
        raise RuntimeError(
            "Class configuration destination changed since it was loaded: "
            f"{target}"
        )
