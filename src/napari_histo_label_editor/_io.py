"""I/O helpers that do not depend on Qt or napari."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import imageio.v3 as iio
import numpy as np
from PIL import Image


PathLike = Union[str, os.PathLike[str]]
_BOX_RESAMPLING = getattr(Image, "Resampling", Image).BOX

__all__ = [
    "FileIdentity",
    "SaveResult",
    "atomic_save_labels",
    "build_image_pyramid",
    "file_identity",
]


@dataclass(frozen=True)
class FileIdentity:
    """Filesystem identity used to detect replacement of a loaded file."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class SaveResult:
    """Canonical destination and identity installed by an atomic save."""

    __slots__ = ("path", "identity")

    path: Path
    identity: FileIdentity


def file_identity(path: PathLike) -> FileIdentity:
    """Return the identity of an absolute file path, following symlinks."""
    target = _canonical_absolute_path(path, "File path")
    return _file_identity_from_status(target.stat())


def _file_identity_from_status(status: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=status.st_dev,
        inode=status.st_ino,
        size=status.st_size,
        mtime_ns=status.st_mtime_ns,
        ctime_ns=status.st_ctime_ns,
    )


def atomic_save_labels(
    labels: np.ndarray,
    destination: PathLike,
    output_dtype: np.dtype | type[np.generic] | str,
    *,
    expected_identity: FileIdentity | None = None,
) -> SaveResult:
    """Write a complete label snapshot and atomically replace *destination*.

    This function is intentionally independent of the UI so it can run in a
    background worker.  Both the dtype conversion and the data copy happen in
    the calling thread.  Thus a worker does not make the GUI thread allocate a
    potentially large on-disk representation, and the writer receives a stable
    snapshot even when the in-memory dtype already matches ``output_dtype``.

    The temporary image is created beside the destination so ``os.replace`` is
    an atomic same-filesystem operation.  A failed conversion, write, flush, or
    replace leaves the existing destination untouched and removes the temporary
    file.  The destination must be absolute after ``~`` expansion. Existing
    symlinks are resolved before writing. The returned ``SaveResult`` contains
    the canonical destination and the identity of the file actually installed.
    When ``expected_identity`` is supplied, the destination must still be the
    exact file represented by that identity both before preparing the snapshot
    and immediately before the atomic replacement.
    """
    # Resolve once, before creating the temporary file.  Besides freezing the
    # destination against later working-directory changes, this follows an
    # existing symlink to its referent.  Replacing the referent keeps the
    # user-selected symlink intact instead of silently replacing the link with
    # a regular file.
    target = _canonical_absolute_path(destination, "Label destination")
    if not target.suffix:
        raise ValueError("Label destination must have an image file extension")

    if expected_identity is not None:
        _verify_file_identity(target, expected_identity)

    target_dtype = np.dtype(output_dtype)
    if target_dtype == np.dtype(np.bool_):
        output_minimum, output_maximum = 0, 1
    elif np.issubdtype(target_dtype, np.integer):
        limits = np.iinfo(target_dtype)
        output_minimum, output_maximum = int(limits.min), int(limits.max)
    else:
        raise ValueError(
            f"Label output dtype must be an integer. Got {target_dtype}"
        )

    source = np.asarray(labels)
    if source.size:
        minimum = int(source.min())
        maximum = int(source.max())
        if minimum < output_minimum or maximum > output_maximum:
            raise ValueError(
                f"Label values from {minimum} to {maximum} cannot be saved "
                f"as {target_dtype} without data loss"
            )

    # Copy even when the dtype already matches.  Label data can remain editable
    # while the worker writes; imageio must see one consistent snapshot.
    snapshot = np.array(source, dtype=target_dtype, copy=True, order="C")

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}.",
        suffix=f".tmp{target.suffix}",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    replacement_complete = False

    try:
        iio.imwrite(temporary_path, snapshot)
        # imageio has closed its handle at this point.  Flush the completed file
        # before making it visible under the destination name.
        with temporary_path.open("rb") as temporary_file:
            os.fsync(temporary_file.fileno())
        temporary_status = temporary_path.stat()
        temporary_device_inode = (
            temporary_status.st_dev,
            temporary_status.st_ino,
        )
        if expected_identity is not None:
            _verify_file_identity(target, expected_identity)
        os.replace(temporary_path, target)
        replacement_complete = True
        try:
            saved_identity = _file_identity_from_status(target.stat())
        except FileNotFoundError as error:
            raise RuntimeError(
                "Label destination changed immediately after the atomic "
                f"save: {target}"
            ) from error
        if (saved_identity.device, saved_identity.inode) != (
            temporary_device_inode
        ):
            raise RuntimeError(
                "Label destination changed immediately after the atomic "
                f"save: {target}"
            )
    except BaseException:
        if not replacement_complete:
            temporary_path.unlink(missing_ok=True)
        raise

    return SaveResult(path=target, identity=saved_identity)


def _canonical_absolute_path(path: PathLike, description: str) -> Path:
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise ValueError(f"{description} must be an absolute path")
    return target.resolve(strict=False)


def _verify_file_identity(
    target: Path,
    expected_identity: FileIdentity,
) -> None:
    try:
        current_identity = file_identity(target)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Label destination no longer exists: {target}"
        ) from error

    if current_identity != expected_identity:
        raise RuntimeError(
            "Label destination changed since it was loaded; refusing to "
            f"overwrite it: {target}"
        )


def build_image_pyramid(
    image: np.ndarray,
    single_scale_limit: int = 8_192,
    downsample: int = 2,
    overview_limit: int = 2_048,
) -> list[np.ndarray]:
    """Return filtered, GPU-friendly levels for an oversized RGB(A) image.

    Images that fit inside ``single_scale_limit`` are returned unchanged. This
    matters for napari 0.6: a multiscale image is cropped and re-uploaded on
    every pan, whereas a single-scale texture remains resident on the GPU.

    Truly oversized images keep level zero at full resolution and receive
    antialiased, C-contiguous overview levels.  Pre-filtering avoids the severe
    color aliasing caused by stride sampling, and compact arrays avoid VisPy's
    expensive gather-copy on every multiscale tile upload.  Levels continue
    until the largest spatial dimension is at most ``overview_limit`` so the
    initial whole-slide view is inexpensive.
    """
    if not isinstance(single_scale_limit, int) or isinstance(
        single_scale_limit,
        bool,
    ):
        raise TypeError("single_scale_limit must be an integer")
    if single_scale_limit < 1:
        raise ValueError("single_scale_limit must be at least 1")
    if not isinstance(downsample, int) or isinstance(downsample, bool):
        raise TypeError("downsample must be an integer")
    if downsample < 2:
        raise ValueError("downsample must be at least 2")
    if not isinstance(overview_limit, int) or isinstance(overview_limit, bool):
        raise TypeError("overview_limit must be an integer")
    if overview_limit < 1:
        raise ValueError("overview_limit must be at least 1")
    if overview_limit >= single_scale_limit:
        raise ValueError(
            "overview_limit must be smaller than single_scale_limit"
        )

    level = np.asarray(image)
    if level.ndim != 3 or level.shape[-1] not in (3, 4):
        raise ValueError(
            "Display pyramid input must be an RGB or RGBA array with shape "
            "(height, width, 3 or 4)"
        )

    pyramid = [level]
    if max(level.shape[:2]) <= single_scale_limit:
        return pyramid

    while max(level.shape[:2]) > overview_limit:
        level = _box_downsample_rgb(level, downsample)
        pyramid.append(level)

    return pyramid


def _box_downsample_rgb(image: np.ndarray, factor: int) -> np.ndarray:
    """Area-filter an RGB(A) level into an owned C-contiguous array.

    Pillow's compiled RGB(A) reducer handles the common uint8 histology path in
    one fast operation. Other dtypes are reduced channel-by-channel so uint16
    and floating-point inputs keep their dtype and numeric range.
    """
    height, width, channels = image.shape
    output_height = (height + factor - 1) // factor
    output_width = (width + factor - 1) // factor
    output_size = (output_width, output_height)

    if image.dtype == np.dtype(np.uint8):
        pil_image = Image.fromarray(image)
        try:
            reduced_image = pil_image.reduce(factor)
        except ValueError:
            reduced_image = pil_image.resize(
                output_size,
                resample=_BOX_RESAMPLING,
            )
        return np.array(
            reduced_image,
            dtype=np.uint8,
            copy=True,
            order="C",
        )

    output = np.empty(
        (output_height, output_width, channels),
        dtype=image.dtype,
        order="C",
    )

    for channel_index in range(channels):
        channel = image[..., channel_index]
        try:
            pil_channel = Image.fromarray(channel)
            try:
                reduced = pil_channel.reduce(factor)
            except ValueError:
                reduced = pil_channel.resize(
                    output_size,
                    resample=_BOX_RESAMPLING,
                )
            reduced_channel = np.asarray(reduced)
        except (KeyError, TypeError, ValueError):
            reduced_channel = _box_downsample_channel_numpy(channel, factor)

        if reduced_channel.shape != (output_height, output_width):
            # Pillow's mode-specific reducer should have the same ceil-sized
            # output, but use an explicit BOX resize if a backend differs.
            reduced_channel = np.asarray(
                Image.fromarray(channel).resize(
                    output_size,
                    resample=_BOX_RESAMPLING,
                )
            )
        output[..., channel_index] = _cast_filtered_channel(
            reduced_channel,
            image.dtype,
        )

    return output


def _box_downsample_channel_numpy(
    channel: np.ndarray,
    factor: int,
) -> np.ndarray:
    """Small-memory typed BOX fallback for Pillow-unsupported channel modes."""
    height, width = channel.shape
    output_height = (height + factor - 1) // factor
    output_width = (width + factor - 1) // factor
    output = np.empty((output_height, output_width), dtype=np.float64)

    for output_row in range(output_height):
        row_start = output_row * factor
        row_stop = min(row_start + factor, height)
        rows = channel[row_start:row_stop]
        full_columns = (width // factor) * factor
        if full_columns:
            blocks = rows[:, :full_columns].reshape(
                row_stop - row_start,
                full_columns // factor,
                factor,
            )
            output[output_row, : full_columns // factor] = blocks.mean(
                axis=(0, 2),
                dtype=np.float64,
            )
        if full_columns < width:
            output[output_row, -1] = rows[:, full_columns:].mean(
                dtype=np.float64
            )

    return output


def _cast_filtered_channel(
    channel: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """Cast a filtered channel back without integer wraparound."""
    target_dtype = np.dtype(dtype)
    if np.dtype(channel.dtype) == target_dtype:
        return channel
    if target_dtype == np.dtype(np.bool_):
        return np.asarray(channel >= 0.5, dtype=target_dtype)
    if np.issubdtype(target_dtype, np.integer):
        limits = np.iinfo(target_dtype)
        channel = np.clip(np.rint(channel), limits.min, limits.max)
    return np.asarray(channel, dtype=target_dtype)
