"""I/O helpers that do not depend on Qt or napari."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import TypeAlias

import imageio.v3 as iio
import numpy as np


PathLike: TypeAlias = str | os.PathLike[str]

__all__ = ["atomic_save_labels", "build_image_pyramid"]


def atomic_save_labels(
    labels: np.ndarray,
    destination: PathLike,
    output_dtype: np.dtype | type[np.generic] | str,
) -> Path:
    """Write a complete label snapshot and atomically replace *destination*.

    This function is intentionally independent of the UI so it can run in a
    background worker.  Both the dtype conversion and the data copy happen in
    the calling thread.  Thus a worker does not make the GUI thread allocate a
    potentially large on-disk representation, and the writer receives a stable
    snapshot even when the in-memory dtype already matches ``output_dtype``.

    The temporary image is created beside the destination so ``os.replace`` is
    an atomic same-filesystem operation.  A failed conversion, write, flush, or
    replace leaves the existing destination untouched and removes the temporary
    file.
    """
    target = Path(destination)
    if not target.suffix:
        raise ValueError("Label destination must have an image file extension")

    # Copy even when the dtype already matches.  Label data can remain editable
    # while the worker writes; imageio must see one consistent snapshot.
    snapshot = np.array(labels, dtype=np.dtype(output_dtype), copy=True, order="C")

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}.",
        suffix=f".tmp{target.suffix}",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        iio.imwrite(temporary_path, snapshot)
        # imageio has closed its handle at this point.  Flush the completed file
        # before making it visible under the destination name.
        with temporary_path.open("rb") as temporary_file:
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return target


def build_image_pyramid(
    image: np.ndarray,
    max_dimension: int = 4096,
    downsample: int = 2,
) -> list[np.ndarray]:
    """Return lightweight multiscale views for a large RGB(A) image.

    Level zero is the original array.  Coarser levels sample every
    ``downsample`` pixels in each spatial dimension without copying data, until
    the largest spatial dimension is at most ``max_dimension``.  This is a
    deliberately conservative display pyramid: it uses no extra large image
    allocations and never changes the source data.

    The stride sampling is suitable for interactive overview rendering, not for
    quantitative resampling or export.
    """
    if not isinstance(max_dimension, int) or isinstance(max_dimension, bool):
        raise TypeError("max_dimension must be an integer")
    if max_dimension < 1:
        raise ValueError("max_dimension must be at least 1")
    if not isinstance(downsample, int) or isinstance(downsample, bool):
        raise TypeError("downsample must be an integer")
    if downsample < 2:
        raise ValueError("downsample must be at least 2")

    level = np.asarray(image)
    if level.ndim != 3 or level.shape[-1] not in (3, 4):
        raise ValueError(
            "Display pyramid input must be an RGB or RGBA array with shape "
            "(height, width, 3 or 4)"
        )

    pyramid = [level]
    while max(level.shape[:2]) > max_dimension:
        level = level[::downsample, ::downsample, :]
        pyramid.append(level)

    return pyramid
