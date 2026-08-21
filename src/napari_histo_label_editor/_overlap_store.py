"""Lossless, packed storage for overlapping semantic annotations.

A conventional two-dimensional label image stores one integer per pixel and
therefore cannot represent two classes at the same location.  ``OverlapStore``
keeps one independent bit plane for every non-background class while exposing
small, binary working planes to napari.  The complete store can be serialized
to a strictly validated byte payload for embedding in the same TIFF or PNG as
the ordinary two-dimensional compatibility projection.

This module deliberately has no Qt or napari imports.  It is safe to use from
background save workers and from format-specific image readers/writers.
"""

from __future__ import annotations

import hashlib
import io
import operator
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._class_config import normalize_color


FORMAT_VERSION = 1
PACK_BITORDER = "little"
_SCAN_BYTES = 8 * 1024 * 1024
_MAX_MOVE_DELTA_BYTES = 64 * 1024 * 1024
_NORMALIZED_COLOR = re.compile(r"#[0-9a-f]{6}\Z")
_GENERATION = re.compile(r"[0-9a-f]{32}\Z")
_REQUIRED_PAYLOAD_KEYS = frozenset(
    {
        "format_version",
        "shape",
        "class_values",
        "class_names",
        "class_colors",
        "background_name",
        "background_color",
        "packed_masks",
        "z_order",
        "projection_dtype",
        "projection_sha256",
        "generation",
    }
)

_BYTE_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, np.newaxis],
    axis=1,
).sum(axis=1, dtype=np.uint8)


__all__ = [
    "FORMAT_VERSION",
    "OverlapDelta",
    "OverlapMoveResult",
    "OverlapStore",
    "hash_projection",
]


def _read_only_copy(values, *, dtype=None) -> np.ndarray:
    """Return an owned C-contiguous array that callers cannot mutate."""

    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class OverlapDelta:
    """Immutable sparse history token for one translated annotation object.

    The before/after arrays are aligned with ``affected_rows`` and
    ``affected_columns``.  They are complete enough to replay either
    direction without unpacking a class plane or reprojecting the slide.
    ``store_id`` intentionally binds a token to its live originating store;
    these runtime history objects are not a serialized file format.
    """

    store_id: int
    shape: tuple[int, int]
    value: int
    row_delta: int
    column_delta: int
    source_rows: np.ndarray
    source_columns: np.ndarray
    destination_rows: np.ndarray
    destination_columns: np.ndarray
    affected_rows: np.ndarray
    affected_columns: np.ndarray
    membership_before: np.ndarray
    membership_after: np.ndarray
    projection_before: np.ndarray
    projection_after: np.ndarray
    source_bounds: tuple[int, int, int, int]
    destination_bounds: tuple[int, int, int, int]
    revision_before: int
    revision_after: int

    @property
    def source_indices(self) -> tuple[np.ndarray, np.ndarray]:
        return self.source_rows, self.source_columns

    @property
    def destination_indices(self) -> tuple[np.ndarray, np.ndarray]:
        return self.destination_rows, self.destination_columns

    @property
    def affected_indices(self) -> tuple[np.ndarray, np.ndarray]:
        return self.affected_rows, self.affected_columns

    @property
    def nbytes(self) -> int:
        """Exact bytes owned by the sparse array payload of this token."""

        arrays = (
            self.source_rows,
            self.source_columns,
            self.destination_rows,
            self.destination_columns,
            self.affected_rows,
            self.affected_columns,
            self.membership_before,
            self.membership_after,
            self.projection_before,
            self.projection_after,
        )
        return sum(int(array.nbytes) for array in arrays)


@dataclass(frozen=True)
class OverlapMoveResult:
    """Bounded refresh information returned after applying a move delta."""

    changed: int
    source_bounds: tuple[int, int, int, int]
    destination_bounds: tuple[int, int, int, int]
    forward: bool
    revision_before: int
    revision_after: int


def _validate_label_array(data, description: str) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"{description} must be a 2D array. Got {array.shape}")
    if not (
        array.dtype == np.dtype(np.bool_)
        or np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(
            f"{description} must have an integer dtype. Got {array.dtype}"
        )
    return array


def _canonical_projection_dtype(dtype) -> np.dtype:
    result = np.dtype(dtype)
    if not (
        result == np.dtype(np.bool_)
        or np.issubdtype(result, np.integer)
    ):
        raise ValueError(
            f"Projection dtype must be an integer dtype. Got {result}"
        )
    if result.itemsize > 1:
        result = result.newbyteorder("<")
    return result


def _projection_hash_prefix(shape: tuple[int, int], dtype: np.dtype) -> bytes:
    return (
        b"napari-histo-overlap-projection-v1\0"
        + np.asarray(shape, dtype="<i8").tobytes()
        + dtype.str.encode("ascii")
        + b"\0"
    )


def hash_projection(projection) -> str:
    """Return a stable SHA-256 digest for one 2-D integer projection.

    Multi-byte values are hashed in little-endian order so an embedded payload
    remains verifiable when a file is moved between machines with different
    native byte orders.  Shape and dtype are included in the digest.
    """

    array = _validate_label_array(projection, "Projection")
    dtype = _canonical_projection_dtype(array.dtype)
    canonical = np.ascontiguousarray(array, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(_projection_hash_prefix(tuple(array.shape), dtype))
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _safe_projection_dtype(dtype, class_values: Iterable[int]) -> np.dtype:
    current = _canonical_projection_dtype(dtype)
    values = tuple(int(value) for value in class_values)
    maximum = max(values, default=0)
    minimum = min((0, *values))

    if current == np.dtype(np.bool_):
        current_minimum, current_maximum = 0, 1
    else:
        limits = np.iinfo(current)
        current_minimum = int(limits.min)
        current_maximum = int(limits.max)
    if current_minimum <= minimum and maximum <= current_maximum:
        return current

    candidates = (
        (np.uint8, np.uint16, np.uint32, np.uint64)
        if minimum >= 0
        else (np.int8, np.int16, np.int32, np.int64)
    )
    for candidate in candidates:
        candidate_dtype = np.dtype(candidate)
        limits = np.iinfo(candidate_dtype)
        if limits.min <= minimum and maximum <= limits.max:
            return _canonical_projection_dtype(candidate_dtype)
    raise ValueError(
        f"Class values from {minimum} to {maximum} exceed integer ranges"
    )


def _normalized_metadata_color(color, description: str) -> str:
    if color is None or str(color).strip() == "":
        return ""
    normalized = normalize_color(color)
    if normalized is None:
        raise ValueError(f"{description} is not a valid color: {color!r}")
    return normalized


def _strict_payload_color(color: str, description: str) -> str:
    if color == "":
        return ""
    if _NORMALIZED_COLOR.fullmatch(color) is None:
        raise ValueError(
            f"{description} must be empty or normalized as #rrggbb"
        )
    return color


def _string_array(values: Iterable[str]) -> np.ndarray:
    strings = tuple(str(value) for value in values)
    width = max((len(value) for value in strings), default=1)
    return np.asarray(strings, dtype=f"<U{width}")


def _coerce_binary_plane(data, shape: tuple[int, int], description: str) -> np.ndarray:
    array = np.asarray(data)
    if array.shape != shape:
        raise ValueError(
            f"{description} must have shape {shape}. Got {array.shape}"
        )
    if array.dtype == np.dtype(np.bool_):
        return array
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{description} must contain only binary values")
    # Validate in bounded row chunks.  Two full-slide comparison masks would
    # otherwise briefly add hundreds of megabytes during Undo/full sync.
    rows_per_chunk = max(
        1,
        min(
            shape[0] or 1,
            _SCAN_BYTES // max(1, shape[1] * array.dtype.itemsize),
        ),
    )
    for row_start in range(0, shape[0], rows_per_chunk):
        chunk = array[row_start : row_start + rows_per_chunk]
        if np.any((chunk != 0) & (chunk != 1)):
            raise ValueError(f"{description} must contain only 0 and 1")
    # ``np.packbits`` accepts both bool and uint8 binary arrays, so retaining
    # the caller's integer view avoids another complete plane allocation.
    return array


def _excluded_values(exclude) -> set[int]:
    if exclude is None:
        return set()
    if isinstance(exclude, (int, np.integer)) and not isinstance(
        exclude, (bool, np.bool_)
    ):
        return {int(exclude)}
    if isinstance(exclude, (str, bytes)):
        raise TypeError("exclude must be a class value or iterable of values")
    try:
        return {int(value) for value in exclude}
    except TypeError as error:
        raise TypeError(
            "exclude must be a class value or iterable of values"
        ) from error


def _copy_projection(projection: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Allocate one independent C-contiguous projection for a transaction."""

    return np.array(projection, dtype=dtype, copy=True, order="C")


class OverlapStore:
    """Packed per-class membership masks plus complete class metadata.

    ``class_values`` contains every configured non-background class, including
    classes whose masks are currently empty.  ``z_order`` is a bottom-to-top
    permutation of those values.  Background is implicit wherever no class is
    present and its name/color are stored separately in the payload.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        class_map: Mapping[int, str],
        packed_masks: np.ndarray,
        *,
        class_colors: Optional[Mapping[int, object]] = None,
        z_order: Optional[Iterable[int]] = None,
        projection_dtype=np.uint8,
        projection_sha256: str = "",
        generation: str = "",
        projection=None,
    ) -> None:
        if len(shape) != 2:
            raise ValueError(f"Overlap shape must have two dimensions: {shape}")
        normalized_shape = tuple(int(size) for size in shape)
        if any(size < 0 for size in normalized_shape):
            raise ValueError("Overlap dimensions cannot be negative")

        normalized_map: dict[int, str] = {}
        for raw_value, raw_name in class_map.items():
            value = int(raw_value)
            if value < 0:
                raise ValueError("Class values cannot be negative")
            name = str(raw_name).strip()
            if not name:
                raise ValueError(f"Class {value} must have a non-blank name")
            normalized_map[value] = name
        normalized_map.setdefault(0, "background")
        class_values = tuple(sorted(value for value in normalized_map if value))

        colors: dict[int, str] = {}
        for raw_value, raw_color in (class_colors or {}).items():
            value = int(raw_value)
            if value not in normalized_map:
                raise ValueError(
                    f"Color is defined for unknown class value {value}"
                )
            color = _normalized_metadata_color(
                raw_color,
                f"Color for class {value}",
            )
            if color:
                colors[value] = color

        packed = np.asarray(packed_masks)
        expected_width = (normalized_shape[1] + 7) // 8
        expected_packed_shape = (
            len(class_values),
            normalized_shape[0],
            expected_width,
        )
        if packed.dtype != np.dtype(np.uint8):
            raise ValueError("Packed masks must have dtype uint8")
        if packed.shape != expected_packed_shape:
            raise ValueError(
                "Packed masks must have shape "
                f"{expected_packed_shape}. Got {packed.shape}"
            )
        packed = np.ascontiguousarray(packed)
        if normalized_shape[1] % 8 and packed.size:
            valid_bits = (1 << (normalized_shape[1] % 8)) - 1
            if np.any(packed[..., -1] & np.uint8(0xFF ^ valid_bits)):
                raise ValueError("Packed masks contain nonzero row-padding bits")

        order = (
            class_values
            if z_order is None
            else tuple(int(value) for value in z_order)
        )
        if len(order) != len(class_values) or set(order) != set(class_values):
            raise ValueError(
                "z_order must contain every non-background class exactly once"
            )
        if len(set(order)) != len(order):
            raise ValueError("z_order cannot contain duplicate class values")

        if projection_sha256 and re.fullmatch(
            r"[0-9a-f]{64}", projection_sha256
        ) is None:
            raise ValueError("projection_sha256 must be 64 lowercase hex digits")
        if generation and _GENERATION.fullmatch(generation) is None:
            raise ValueError("generation must be 32 lowercase hex digits")

        self._shape = normalized_shape
        self._class_map = normalized_map
        self._class_colors = colors
        self._class_values = class_values
        self._value_to_index = {
            value: index for index, value in enumerate(class_values)
        }
        self._packed_masks = packed
        self._z_order = order
        # Runtime projection memory depends on class values, not the dtype of
        # the outer PNG/TIFF. A legacy int32 mask with values 0..22 therefore
        # stays uint8 in RAM; save can cast the shared top projection back to
        # the original on-disk dtype without changing the embedded hash.
        _safe_projection_dtype(projection_dtype, class_values)
        self._projection_dtype = _safe_projection_dtype(
            np.uint8,
            class_values,
        )
        self._projection_sha256 = str(projection_sha256)
        self._generation = str(generation)
        self._revision = 0
        if projection is None:
            self._projection = self._project_by_global_order(
                dtype=self._projection_dtype,
            )
        else:
            projection_array = _validate_label_array(
                projection,
                "Projection",
            )
            if tuple(projection_array.shape) != self._shape:
                raise ValueError(
                    f"Projection shape {projection_array.shape} does not "
                    f"match overlap shape {self._shape}"
                )
            # Validate the original values before compacting. Otherwise a
            # malicious or corrupt outer projection such as uint16 value 257
            # could wrap to class 1 in uint8 and appear membership-consistent.
            if not self._projection_memberships_valid(projection_array):
                raise ValueError(
                    "Projection top values do not match overlap memberships"
                )
            self._projection = np.array(
                projection_array,
                dtype=self._projection_dtype,
                copy=True,
                order="C",
            )

    @classmethod
    def from_legacy(
        cls,
        labels,
        class_map: Optional[Mapping[int, str] | Iterable[int]] = None,
        class_colors: Optional[Mapping[int, object]] = None,
        z_order: Optional[Iterable[int]] = None,
    ) -> "OverlapStore":
        """Losslessly migrate one categorical 2-D mask to one-hot planes."""

        data = _validate_label_array(labels, "Legacy labels")
        if data.size and int(data.min()) < 0:
            raise ValueError("Legacy labels cannot contain negative class values")

        observed: set[int] = set()
        rows_per_chunk = max(
            1,
            min(
                data.shape[0] or 1,
                _SCAN_BYTES // max(1, data.shape[1] * data.dtype.itemsize),
            ),
        )
        for row_start in range(0, data.shape[0], rows_per_chunk):
            observed.update(
                int(value)
                for value in np.unique(
                    data[row_start : row_start + rows_per_chunk]
                )
                if int(value) != 0
            )

        if class_map is None:
            metadata = {value: f"Class {value}" for value in observed}
        elif isinstance(class_map, Mapping):
            metadata = {
                int(value): str(name) for value, name in class_map.items()
            }
        else:
            metadata = {
                int(value): f"Class {int(value)}" for value in class_map
            }
        metadata.setdefault(0, "background")
        for value in observed:
            metadata.setdefault(value, f"Class {value}")
        if any(value < 0 for value in metadata):
            raise ValueError("Class values cannot be negative")

        values = tuple(sorted(value for value in metadata if value))
        height, width = data.shape
        packed = np.zeros(
            (len(values), height, (width + 7) // 8),
            dtype=np.uint8,
        )
        mask_rows = max(1, min(height or 1, _SCAN_BYTES // max(1, width)))
        for class_index, value in enumerate(values):
            for row_start in range(0, height, mask_rows):
                row_stop = min(height, row_start + mask_rows)
                matches = data[row_start:row_stop] == value
                packed[class_index, row_start:row_stop] = np.packbits(
                    matches,
                    axis=-1,
                    bitorder=PACK_BITORDER,
                )

        result = cls(
            data.shape,
            metadata,
            packed,
            class_colors=class_colors,
            z_order=z_order,
            projection_dtype=data.dtype,
            projection_sha256=hash_projection(data),
            generation=uuid.uuid4().hex,
            projection=data,
        )
        return result

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def class_values(self) -> tuple[int, ...]:
        return self._class_values

    @property
    def class_map(self) -> dict[int, str]:
        return dict(self._class_map)

    @property
    def class_colors(self) -> dict[int, str]:
        return dict(self._class_colors)

    @property
    def z_order(self) -> tuple[int, ...]:
        return self._z_order

    @property
    def projection_dtype(self) -> np.dtype:
        return self._projection_dtype

    @property
    def projection_sha256(self) -> str:
        return self._projection_sha256

    @property
    def projection_view(self) -> np.ndarray:
        """Return a stable read-only view of the authoritative top labels."""

        view = self._projection.view()
        view.setflags(write=False)
        return view

    @property
    def generation(self) -> str:
        return self._generation

    @property
    def revision(self) -> int:
        """Monotonic in-memory mutation counter for stale-selection checks."""

        return self._revision

    @property
    def packed_masks(self) -> np.ndarray:
        """Return a read-only view of the packed ``(C, H, ceil(W/8))`` data."""

        view = self._packed_masks.view()
        view.setflags(write=False)
        return view

    def _require_class(self, value: int) -> int:
        value = int(value)
        try:
            return self._value_to_index[value]
        except KeyError as error:
            raise ValueError(f"Unknown non-background class value {value}") from error

    def select_plane(self, value: int) -> np.ndarray:
        """Return one independent writable boolean membership plane."""

        index = self._require_class(value)
        return np.unpackbits(
            self._packed_masks[index],
            axis=-1,
            count=self._shape[1],
            bitorder=PACK_BITORDER,
        ).view(np.bool_)

    # ``unpack_plane`` is a descriptive alias useful outside the UI.
    unpack_plane = select_plane

    def update_plane(self, value: int, plane) -> int:
        """Replace one class plane and return the number of changed pixels."""

        changed, _added, _removed = self.update_plane_counts(value, plane)
        return changed

    def update_plane_counts(self, value: int, plane) -> tuple[int, int, int]:
        """Replace one plane and return ``(changed, added, removed)`` counts.

        Counts are computed from packed bytes, avoiding the multiple
        slide-sized boolean temporaries that an Undo/Redo full sync would
        otherwise require.
        """

        index = self._require_class(value)
        binary = _coerce_binary_plane(plane, self._shape, "Class plane")
        replacement = np.packbits(
            binary,
            axis=-1,
            bitorder=PACK_BITORDER,
        )
        difference = np.bitwise_xor(self._packed_masks[index], replacement)
        changed = int(_BYTE_POPCOUNT[difference].sum(dtype=np.uint64))
        added_bits = np.bitwise_and(difference, replacement)
        added = int(_BYTE_POPCOUNT[added_bits].sum(dtype=np.uint64))
        removed = changed - added
        self._packed_masks[index] = replacement
        if changed:
            height, width = self._shape
            rows_per_chunk = max(
                1,
                min(height or 1, _SCAN_BYTES // max(1, width * 3)),
            )
            for row_start in range(0, height, rows_per_chunk):
                row_stop = min(height, row_start + rows_per_chunk)
                added_mask = np.unpackbits(
                    added_bits[row_start:row_stop],
                    axis=-1,
                    count=width,
                    bitorder=PACK_BITORDER,
                ).view(np.bool_)
                removed_mask = np.unpackbits(
                    np.bitwise_and(
                        difference[row_start:row_stop],
                        self._packed_masks[index, row_start:row_stop]
                        ^ difference[row_start:row_stop],
                    ),
                    axis=-1,
                    count=width,
                    bitorder=PACK_BITORDER,
                ).view(np.bool_)
                # ``difference & old`` above identifies 1 -> 0 removals after
                # the packed plane has already been replaced.
                projection_chunk = self._projection[row_start:row_stop]
                projection_chunk[added_mask] = value
                reveal = removed_mask & (projection_chunk == value)
                if np.any(reveal):
                    self._reveal_patch(
                        projection_chunk,
                        reveal,
                        (row_start, 0),
                        exclude=value,
                    )
            self._mark_modified()
        return changed, added, removed

    def _validate_patch_bounds(
        self,
        offset,
        patch_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        try:
            row, column = offset
        except (TypeError, ValueError) as error:
            raise ValueError("Patch offset must contain row and column") from error
        if isinstance(row, (bool, np.bool_)) or isinstance(
            column, (bool, np.bool_)
        ):
            raise ValueError("Patch offset must contain integer coordinates")
        row, column = int(row), int(column)
        row_stop = row + int(patch_shape[0])
        column_stop = column + int(patch_shape[1])
        if (
            row < 0
            or column < 0
            or row_stop > self._shape[0]
            or column_stop > self._shape[1]
        ):
            raise ValueError(
                f"Patch at {(row, column)} with shape {patch_shape} exceeds "
                f"store shape {self._shape}"
            )
        return row, column, row_stop, column_stop

    def update_patch(self, value: int, offset, patch) -> int:
        """Set a rectangular membership patch and return changed-pixel count."""

        index = self._require_class(value)
        patch_array = np.asarray(patch)
        if patch_array.ndim != 2:
            raise ValueError("Membership patch must be 2D")
        binary = _coerce_binary_plane(
            patch_array,
            tuple(patch_array.shape),
            "Membership patch",
        )
        row, column, row_stop, column_stop = self._validate_patch_bounds(
            offset,
            tuple(binary.shape),
        )
        if binary.size == 0:
            return 0

        byte_start = column // 8
        byte_stop = (column_stop + 7) // 8
        bit_start = column - byte_start * 8
        packed_block = self._packed_masks[
            index,
            row:row_stop,
            byte_start:byte_stop,
        ]
        unpacked = np.unpackbits(
            packed_block,
            axis=-1,
            bitorder=PACK_BITORDER,
        )
        target = unpacked[:, bit_start : bit_start + binary.shape[1]]
        added_mask = (target == 0) & (binary != 0)
        removed_mask = (target != 0) & (binary == 0)
        changed = int(np.count_nonzero(added_mask) + np.count_nonzero(removed_mask))
        target[...] = binary
        self._packed_masks[
            index,
            row:row_stop,
            byte_start:byte_stop,
        ] = np.packbits(unpacked, axis=-1, bitorder=PACK_BITORDER)
        if changed:
            projection_patch = self._projection[
                row:row_stop,
                column:column_stop,
            ]
            projection_patch[added_mask] = value
            reveal = removed_mask & (projection_patch == value)
            if np.any(reveal):
                self._reveal_patch(
                    projection_patch,
                    reveal,
                    (row, column),
                    exclude=value,
                )
            self._mark_modified()
        return changed

    def _plane_patch(
        self,
        value: int,
        offset,
        shape: tuple[int, int],
    ) -> np.ndarray:
        index = self._require_class(value)
        row, column, row_stop, column_stop = self._validate_patch_bounds(
            offset,
            shape,
        )
        if shape[0] == 0 or shape[1] == 0:
            return np.zeros(shape, dtype=bool)
        byte_start = column // 8
        byte_stop = (column_stop + 7) // 8
        bit_start = column - byte_start * 8
        unpacked = np.unpackbits(
            self._packed_masks[
                index,
                row:row_stop,
                byte_start:byte_stop,
            ],
            axis=-1,
            bitorder=PACK_BITORDER,
        )
        return unpacked[:, bit_start : bit_start + shape[1]].view(np.bool_)

    def select_patch(self, value: int, offset, shape) -> np.ndarray:
        """Return an independent boolean membership rectangle.

        This lets an editing controller compare a small incoming napari update
        with its prior membership bits, without unpacking a slide-sized plane.
        """

        try:
            patch_shape = tuple(int(size) for size in shape)
        except TypeError as error:
            raise ValueError(
                "Membership patch shape must have two dimensions"
            ) from error
        if len(patch_shape) != 2 or any(size < 0 for size in patch_shape):
            raise ValueError(
                "Membership patch shape must have two nonnegative dimensions"
            )
        return self._plane_patch(value, offset, patch_shape)

    def memberships_at(self, row: int, column: int) -> tuple[int, ...]:
        """Return memberships with the authoritative visible class last."""

        if isinstance(row, (bool, np.bool_)) or isinstance(
            column, (bool, np.bool_)
        ):
            raise ValueError("Pixel coordinates must be integers")
        row, column = int(row), int(column)
        if not (0 <= row < self._shape[0] and 0 <= column < self._shape[1]):
            raise IndexError(
                f"Pixel {(row, column)} is outside store shape {self._shape}"
            )
        byte = column // 8
        bit = np.uint8(1 << (column % 8))
        memberships = [
            value
            for value in self._z_order
            if self._packed_masks[self._value_to_index[value], row, byte] & bit
        ]
        visible = int(self._projection[row, column])
        if visible in memberships:
            memberships.remove(visible)
            memberships.append(visible)
        return tuple(memberships)

    def raise_projection_indices(self, value: int, indices) -> int:
        """Make an existing membership visible at sparse requested pixels.

        Membership planes are unchanged. Only coordinates that already
        belong to ``value`` are raised, so repainting a hidden class is an
        idempotent local operation and cannot affect unrelated overlaps.
        """

        value = int(value)
        self._require_class(value)
        rows, columns = self._sparse_indices(indices)
        if rows.size == 0:
            return 0
        membership = self._membership_at_indices(value, rows, columns)
        changed = membership & (self._projection[rows, columns] != value)
        if not np.any(changed):
            return 0
        self._projection[rows[changed], columns[changed]] = value
        count = int(np.count_nonzero(changed))
        self._mark_modified()
        return count

    def plan_move(
        self,
        value: int,
        indices,
        offset,
        *,
        expected_revision: int | None = None,
    ) -> OverlapDelta:
        """Plan an exact sparse translation without mutating the store.

        The selected source must still be visibly owned by ``value``.  The
        resulting membership is ``(old & ~source) | destination``; translated
        pixels become visibly topmost while source-only pixels reveal the
        stable hidden fallback. A destination cannot contain a same-class
        membership outside the source because one boolean class plane cannot
        preserve two object identities after such a merge. ``expected_revision``
        should be the revision captured when the UI selected the object.
        """

        value = int(value)
        self._require_class(value)
        if expected_revision is not None:
            try:
                selection_revision = operator.index(expected_revision)
            except TypeError as error:
                raise TypeError("Expected revision must be an integer") from error
            if selection_revision != self._revision:
                raise RuntimeError(
                    "The selected annotation is stale; select it again."
                )

        try:
            raw_row_delta, raw_column_delta = offset
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Move offset must contain row and column deltas"
            ) from error
        if isinstance(raw_row_delta, (bool, np.bool_)) or isinstance(
            raw_column_delta,
            (bool, np.bool_),
        ):
            raise TypeError("Move deltas must be integers")
        try:
            row_delta = operator.index(raw_row_delta)
            column_delta = operator.index(raw_column_delta)
        except TypeError as error:
            raise TypeError("Move deltas must be integers") from error
        if row_delta == 0 and column_delta == 0:
            raise ValueError("Move offset cannot be zero")

        source_rows, source_columns = self._sparse_indices(indices)
        if source_rows.size == 0:
            raise ValueError("A moved annotation cannot be empty")
        width = self._shape[1]
        source_linear = (
            source_rows.astype(np.intp, copy=False) * width
            + source_columns.astype(np.intp, copy=False)
        )
        if np.unique(source_linear).size != source_linear.size:
            raise ValueError("Moved annotation coordinates must be unique")

        row_min = int(source_rows.min()) + row_delta
        row_max = int(source_rows.max()) + row_delta
        column_min = int(source_columns.min()) + column_delta
        column_max = int(source_columns.max()) + column_delta
        if (
            row_min < 0
            or column_min < 0
            or row_max >= self._shape[0]
            or column_max >= self._shape[1]
        ):
            raise ValueError("The moved annotation would leave the image")
        if np.any(self._projection[source_rows, source_columns] != value):
            raise RuntimeError(
                "The selected annotation is stale; select it again."
            )

        # Refuse an operation whose immutable history token could itself put
        # the application under memory pressure. This check precedes all
        # destination/union allocations and intentionally overestimates a
        # maximally disjoint source and destination.
        source_count = int(source_rows.size)
        affected_limit = source_count * 2
        estimated_token_bytes = (
            source_count * np.dtype(np.intp).itemsize * 4
            + affected_limit * np.dtype(np.intp).itemsize * 2
            + affected_limit * 2
            + affected_limit * self._projection.dtype.itemsize * 2
        )
        if estimated_token_bytes > _MAX_MOVE_DELTA_BYTES:
            raise ValueError(
                "The selected annotation is too large to move safely; "
                "split it into smaller annotations first."
            )

        source_rows = np.asarray(source_rows, dtype=np.intp)
        source_columns = np.asarray(source_columns, dtype=np.intp)
        destination_rows = source_rows + row_delta
        destination_columns = source_columns + column_delta
        destination_linear = destination_rows * width + destination_columns
        affected_linear = np.unique(
            np.concatenate((source_linear, destination_linear))
        )
        affected_rows = affected_linear // width
        affected_columns = affected_linear % width
        source_positions = np.searchsorted(affected_linear, source_linear)
        destination_positions = np.searchsorted(
            affected_linear,
            destination_linear,
        )

        membership_before = self._membership_at_indices(
            value,
            affected_rows,
            affected_columns,
        )
        source_membership = np.zeros(affected_rows.shape, dtype=bool)
        source_membership[source_positions] = True
        destination_outside_source = ~source_membership[
            destination_positions
        ]
        if np.any(
            membership_before[
                destination_positions[destination_outside_source]
            ]
        ):
            raise ValueError(
                "The move would merge with another annotation of the same "
                "class. Move somewhere that does not overlap that class."
            )
        membership_after = np.array(membership_before, copy=True)
        membership_after[source_positions] = False
        membership_after[destination_positions] = True

        projection_before = np.array(
            self._projection[affected_rows, affected_columns],
            copy=True,
        )
        projection_after = np.zeros(
            affected_rows.shape,
            dtype=self._projection.dtype,
        )
        for class_value in self._z_order:
            membership = (
                membership_after
                if class_value == value
                else self._membership_at_indices(
                    class_value,
                    affected_rows,
                    affected_columns,
                )
            )
            projection_after[membership] = class_value
        # A move is a local paint at the destination, independent of global
        # class order or any previously raised top at those pixels.
        projection_after[destination_positions] = value

        source_bounds = (
            int(source_rows.min()),
            int(source_rows.max()) + 1,
            int(source_columns.min()),
            int(source_columns.max()) + 1,
        )
        destination_bounds = (
            int(destination_rows.min()),
            int(destination_rows.max()) + 1,
            int(destination_columns.min()),
            int(destination_columns.max()) + 1,
        )
        revision_before = self._revision
        return OverlapDelta(
            store_id=id(self),
            shape=self._shape,
            value=value,
            row_delta=row_delta,
            column_delta=column_delta,
            source_rows=_read_only_copy(source_rows, dtype=np.intp),
            source_columns=_read_only_copy(source_columns, dtype=np.intp),
            destination_rows=_read_only_copy(
                destination_rows,
                dtype=np.intp,
            ),
            destination_columns=_read_only_copy(
                destination_columns,
                dtype=np.intp,
            ),
            affected_rows=_read_only_copy(affected_rows, dtype=np.intp),
            affected_columns=_read_only_copy(
                affected_columns,
                dtype=np.intp,
            ),
            membership_before=_read_only_copy(
                membership_before,
                dtype=bool,
            ),
            membership_after=_read_only_copy(
                membership_after,
                dtype=bool,
            ),
            projection_before=_read_only_copy(
                projection_before,
                dtype=self._projection.dtype,
            ),
            projection_after=_read_only_copy(
                projection_after,
                dtype=self._projection.dtype,
            ),
            source_bounds=source_bounds,
            destination_bounds=destination_bounds,
            revision_before=revision_before,
            revision_after=revision_before + 1,
        )

    def apply_delta(
        self,
        delta: OverlapDelta,
        *,
        forward: bool,
        expected_revision: int | None = None,
    ) -> OverlapMoveResult:
        """Atomically apply or reverse one :class:`OverlapDelta`.

        Both the packed membership and authoritative sparse top values must
        match the requested side of the token. A stale or out-of-order token
        is rejected before mutation. Failed packed writes restore raw bytes,
        projection values, metadata hashes, and the in-memory revision.
        """

        if not isinstance(delta, OverlapDelta):
            raise TypeError("delta must be an OverlapDelta")
        if not isinstance(forward, (bool, np.bool_)):
            raise TypeError("forward must be a boolean")
        if delta.store_id != id(self) or tuple(delta.shape) != self._shape:
            raise ValueError("Move delta belongs to a different overlap store")
        self._require_class(delta.value)
        if expected_revision is not None:
            try:
                required_revision = operator.index(expected_revision)
            except TypeError as error:
                raise TypeError("Expected revision must be an integer") from error
            if required_revision != self._revision:
                raise RuntimeError("Move history revision is stale")

        rows, columns = self._sparse_indices(delta.affected_indices)
        expected_membership = np.asarray(
            delta.membership_before
            if forward
            else delta.membership_after
        )
        desired_membership = np.asarray(
            delta.membership_after
            if forward
            else delta.membership_before
        )
        expected_projection = np.asarray(
            delta.projection_before
            if forward
            else delta.projection_after
        )
        desired_projection = np.asarray(
            delta.projection_after
            if forward
            else delta.projection_before
        )
        if any(array.shape != rows.shape for array in (
            expected_membership,
            desired_membership,
            expected_projection,
            desired_projection,
        )) or rows.shape != columns.shape:
            raise ValueError("Move delta arrays are not aligned")
        if (
            expected_membership.dtype != np.dtype(np.bool_)
            or desired_membership.dtype != np.dtype(np.bool_)
        ):
            raise TypeError("Move membership history must be boolean")
        if not (
            np.issubdtype(expected_projection.dtype, np.integer)
            and np.issubdtype(desired_projection.dtype, np.integer)
        ):
            raise TypeError("Move projection history must be integer")
        if rows.size:
            linear = rows * self._shape[1] + columns
            if np.unique(linear).size != linear.size:
                raise ValueError("Move delta coordinates must be unique")

        current_membership = self._membership_at_indices(
            delta.value,
            rows,
            columns,
        )
        current_projection = np.array(
            self._projection[rows, columns],
            copy=True,
        )
        if not np.array_equal(current_membership, expected_membership) or not (
            np.array_equal(current_projection, expected_projection)
        ):
            raise RuntimeError("Move delta no longer matches overlap state")

        # Validate the complete requested sparse projection before touching a
        # packed byte. This also protects the store if a caller fabricates or
        # tampers with a runtime history token.
        occupied = np.zeros(rows.shape, dtype=bool)
        valid_top = np.zeros(rows.shape, dtype=bool)
        for class_value in self._class_values:
            membership = (
                desired_membership
                if class_value == delta.value
                else self._membership_at_indices(
                    class_value,
                    rows,
                    columns,
                )
            )
            occupied |= membership
            valid_top |= membership & (desired_projection == class_value)
        if np.any((desired_projection == 0) != ~occupied) or np.any(
            (desired_projection != 0) & ~valid_top
        ):
            raise ValueError(
                "Move projection history does not match memberships"
            )

        changed = (current_membership != desired_membership) | (
            current_projection != desired_projection
        )
        revision_before = self._revision
        changed_count = int(np.count_nonzero(changed))
        result = OverlapMoveResult(
            changed=changed_count,
            source_bounds=delta.source_bounds,
            destination_bounds=delta.destination_bounds,
            forward=bool(forward),
            revision_before=revision_before,
            revision_after=revision_before + int(changed_count != 0),
        )
        if changed_count == 0:
            return result

        selector_values = np.broadcast_to(
            np.asarray(delta.value, dtype=np.int64),
            rows.shape,
        )
        packed_snapshot = self._snapshot_membership_bytes(
            (delta.value,),
            rows,
            columns,
            selector_values=selector_values,
        )
        projection_sha256 = self._projection_sha256
        generation = self._generation
        revision = self._revision
        try:
            removals = current_membership & ~desired_membership
            additions = ~current_membership & desired_membership
            self._set_membership_at_indices(
                delta.value,
                rows[removals],
                columns[removals],
                present=False,
            )
            self._set_membership_at_indices(
                delta.value,
                rows[additions],
                columns[additions],
                present=True,
            )
            self._projection[rows, columns] = desired_projection
            self._mark_modified()
        except BaseException:
            self._restore_membership_bytes(packed_snapshot)
            self._projection[rows, columns] = current_projection
            self._projection_sha256 = projection_sha256
            self._generation = generation
            self._revision = revision
            raise
        return result

    def erase_visible_indices(
        self,
        indices,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Remove the currently visible membership at sparse coordinates.

        Each non-background pixel may have a different visible class.  Only
        that top membership is removed; hidden memberships are preserved and
        the stable class order supplies the newly revealed top. Coordinates
        must be unique; the controller collapses duplicates so one callback
        cannot drill through multiple memberships at the same pixel.

        Returns
        -------
        changed : int
            Number of visible memberships removed.
        top_before, top_after : numpy.ndarray
            Sparse projection values aligned with ``indices``.
        """

        rows, columns = self._sparse_indices(indices)
        if rows.size:
            linear = rows.astype(np.intp, copy=False) * self._shape[1]
            linear = linear + columns.astype(np.intp, copy=False)
            if np.unique(linear).size != linear.size:
                raise ValueError("Semantic eraser indices must be unique")
        top_before = np.array(
            self._projection[rows, columns],
            copy=True,
        )
        top_after = top_before.copy()
        erase = top_before != 0
        if not np.any(erase):
            return 0, top_before, top_after

        top_after[erase] = 0
        # z_order is bottom-to-top, so later present memberships overwrite
        # earlier ones and become the stable revealed fallback. Compute the
        # complete candidate before mutating packed bits so an allocation
        # failure cannot leave memberships and projection inconsistent.
        for value in self._z_order:
            membership = self._membership_at_indices(value, rows, columns)
            membership &= ~(erase & (top_before == value))
            top_after[erase & membership] = value
        affected_values = np.unique(top_before[erase])
        packed_snapshot = self._snapshot_membership_bytes(
            affected_values,
            rows,
            columns,
            selector_values=top_before,
        )
        projection_sha256 = self._projection_sha256
        generation = self._generation
        revision = self._revision
        try:
            for raw_value in affected_values:
                value = int(raw_value)
                selected = erase & (top_before == value)
                self._set_membership_at_indices(
                    value,
                    rows[selected],
                    columns[selected],
                    present=False,
                )
            self._projection[rows[erase], columns[erase]] = top_after[erase]
            self._mark_modified()
        except BaseException:
            self._restore_membership_bytes(packed_snapshot)
            self._projection[rows, columns] = top_before
            self._projection_sha256 = projection_sha256
            self._generation = generation
            self._revision = revision
            raise
        return int(np.count_nonzero(erase)), top_before, top_after

    def restore_erased_indices(
        self,
        indices,
        erased_values,
        projection_values,
        *,
        present: bool,
    ) -> int:
        """Restore or replay sparse semantic eraser history exactly.

        ``present=True`` re-adds the erased memberships for Undo;
        ``present=False`` removes them for Redo.  The requested projection is
        validated against the resulting memberships before any packed bits or
        visible values are changed.
        """

        if not isinstance(present, (bool, np.bool_)):
            raise TypeError("present must be a boolean")
        rows, columns = self._sparse_indices(indices)
        if rows.size:
            linear = rows.astype(np.intp, copy=False) * self._shape[1]
            linear = linear + columns.astype(np.intp, copy=False)
            if np.unique(linear).size != linear.size:
                raise ValueError("Semantic eraser history indices must be unique")

        erased = self._sparse_history_values(
            erased_values,
            rows.shape,
            "Erased membership values",
        )
        requested = self._sparse_history_values(
            projection_values,
            rows.shape,
            "Projection history values",
        )
        for raw_value in np.unique(erased):
            value = int(raw_value)
            if value != 0:
                self._require_class(value)
        if rows.size == 0:
            return 0

        occupied = np.zeros(rows.shape, dtype=bool)
        valid_top = np.zeros(rows.shape, dtype=bool)
        membership_changed = np.zeros(rows.shape, dtype=bool)
        for value in self._class_values:
            membership = self._membership_at_indices(
                value,
                rows,
                columns,
            )
            affected = erased == value
            if np.any(affected):
                membership_changed[affected] = (
                    membership[affected] != bool(present)
                )
                membership = membership.copy()
                membership[affected] = bool(present)
            occupied |= membership
            valid_top |= membership & (requested == value)
        if np.any((requested == 0) != ~occupied) or np.any(
            (requested != 0) & ~valid_top
        ):
            raise ValueError(
                "Semantic eraser history projection does not match "
                "memberships"
            )

        affected_values = np.unique(erased[erased != 0])
        packed_snapshot = self._snapshot_membership_bytes(
            affected_values,
            rows,
            columns,
            selector_values=erased,
        )
        projection_before = np.array(
            self._projection[rows, columns],
            copy=True,
        )
        projection_sha256 = self._projection_sha256
        generation = self._generation
        revision = self._revision
        try:
            for raw_value in affected_values:
                value = int(raw_value)
                selected = erased == value
                self._set_membership_at_indices(
                    value,
                    rows[selected],
                    columns[selected],
                    present=bool(present),
                )
            projection_changed = projection_before != requested
            self._projection[rows, columns] = requested
            changed = membership_changed | projection_changed
            if np.any(changed):
                self._mark_modified()
        except BaseException:
            self._restore_membership_bytes(packed_snapshot)
            self._projection[rows, columns] = projection_before
            self._projection_sha256 = projection_sha256
            self._generation = generation
            self._revision = revision
            raise
        return int(np.count_nonzero(changed))

    def snapshot_erased_transaction(self, changes):
        """Capture sparse raw state for an atomic multi-atom history action.

        ``changes`` contains ``(indices, erased_values)`` pairs. The opaque
        return value owns only affected packed bytes and projection pixels;
        allocation completes before any Undo/Redo atom mutates the store.
        """

        packed_snapshot = []
        projection_snapshot = []
        for indices, erased_values in changes:
            rows, columns = self._sparse_indices(indices)
            erased = self._sparse_history_values(
                erased_values,
                rows.shape,
                "Erased membership values",
            )
            affected_values = np.unique(erased[erased != 0])
            packed_snapshot.extend(
                self._snapshot_membership_bytes(
                    affected_values,
                    rows,
                    columns,
                    selector_values=erased,
                )
            )
            projection_snapshot.append(
                (
                    rows,
                    columns,
                    np.array(self._projection[rows, columns], copy=True),
                )
            )
        return (
            packed_snapshot,
            projection_snapshot,
            self._projection_sha256,
            self._generation,
            self._revision,
        )

    def restore_erased_transaction(self, snapshot) -> None:
        """Restore an opaque multi-atom snapshot without new allocations."""

        packed, projection, projection_sha256, generation, revision = snapshot
        self._restore_membership_bytes(packed)
        for rows, columns, values in projection:
            self._projection[rows, columns] = values
        self._projection_sha256 = projection_sha256
        self._generation = generation
        self._revision = revision

    def restore_projection_indices(self, indices, values) -> int:
        """Restore sparse authoritative tops after membership Undo/Redo.

        The requested values must describe a valid projection for the current
        membership planes. This prevents history corruption from creating a
        visible class that is not actually present at a pixel.
        """

        rows, columns = self._sparse_indices(indices)
        requested = np.asarray(values)
        if requested.ndim == 0:
            requested = np.full(rows.shape, requested, dtype=requested.dtype)
        else:
            try:
                requested = np.broadcast_to(requested, rows.shape)
            except ValueError as error:
                raise ValueError(
                    "Projection history values must match sparse indices"
                ) from error
        if not (
            requested.dtype == np.dtype(np.bool_)
            or np.issubdtype(requested.dtype, np.integer)
        ):
            raise TypeError("Projection history values must be integers")
        if rows.size == 0:
            return 0

        occupied = np.zeros(rows.shape, dtype=bool)
        valid_top = np.zeros(rows.shape, dtype=bool)
        for class_value in self._class_values:
            membership = self._membership_at_indices(
                class_value,
                rows,
                columns,
            )
            occupied |= membership
            valid_top |= membership & (requested == class_value)
        if np.any((requested == 0) != ~occupied) or np.any(
            (requested != 0) & ~valid_top
        ):
            raise ValueError(
                "Projection history values do not match overlap memberships"
            )

        changed = self._projection[rows, columns] != requested
        if not np.any(changed):
            return 0
        self._projection[rows[changed], columns[changed]] = requested[changed]
        count = int(np.count_nonzero(changed))
        self._mark_modified()
        return count

    def _sparse_indices(self, indices) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(indices, tuple) or len(indices) != 2:
            raise ValueError("Sparse indices must be a (rows, columns) tuple")
        rows, columns = np.broadcast_arrays(
            np.asarray(indices[0]),
            np.asarray(indices[1]),
        )
        if not (
            np.issubdtype(rows.dtype, np.integer)
            and np.issubdtype(columns.dtype, np.integer)
        ):
            raise TypeError("Sparse indices must be integers")
        rows = rows.reshape(-1)
        columns = columns.reshape(-1)
        if rows.size and (
            int(rows.min()) < 0
            or int(columns.min()) < 0
            or int(rows.max()) >= self._shape[0]
            or int(columns.max()) >= self._shape[1]
        ):
            raise IndexError("Sparse indices exceed overlap shape")
        return rows, columns

    def _membership_at_indices(
        self,
        value: int,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> np.ndarray:
        plane_index = self._value_to_index[int(value)]
        packed = self._packed_masks[
            plane_index,
            rows,
            columns // 8,
        ]
        bits = np.left_shift(np.uint8(1), (columns % 8).astype(np.uint8))
        return (packed & bits) != 0

    def _set_membership_at_indices(
        self,
        value: int,
        rows: np.ndarray,
        columns: np.ndarray,
        *,
        present: bool,
    ) -> None:
        """Set sparse packed membership bits, accumulating shared bytes."""

        if rows.size == 0:
            return
        plane = self._packed_masks[self._require_class(value)]
        byte_columns = columns // 8
        bits = np.left_shift(np.uint8(1), (columns % 8).astype(np.uint8))
        if present:
            np.bitwise_or.at(plane, (rows, byte_columns), bits)
        else:
            np.bitwise_and.at(
                plane,
                (rows, byte_columns),
                np.bitwise_not(bits),
            )

    def _snapshot_membership_bytes(
        self,
        values,
        rows: np.ndarray,
        columns: np.ndarray,
        *,
        selector_values: np.ndarray,
    ) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """Copy only packed bytes touched by a sparse multi-class change."""

        snapshot = []
        packed_width = self._packed_masks.shape[2]
        for raw_value in values:
            value = int(raw_value)
            selected = selector_values == value
            selected_rows = rows[selected]
            byte_columns = columns[selected] // 8
            flat_bytes = selected_rows * packed_width + byte_columns
            unique_bytes = np.unique(flat_bytes)
            selected_rows = unique_bytes // packed_width
            byte_columns = unique_bytes % packed_width
            plane_index = self._require_class(value)
            packed_values = np.array(
                self._packed_masks[
                    plane_index,
                    selected_rows,
                    byte_columns,
                ],
                copy=True,
            )
            snapshot.append(
                (
                    plane_index,
                    selected_rows,
                    byte_columns,
                    packed_values,
                )
            )
        return snapshot

    def _restore_membership_bytes(
        self,
        snapshot: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        """Restore a sparse packed-byte snapshot without public setters."""

        for plane_index, rows, byte_columns, packed_values in snapshot:
            self._packed_masks[
                plane_index,
                rows,
                byte_columns,
            ] = packed_values

    @staticmethod
    def _sparse_history_values(values, shape, description: str) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == 0:
            array = np.full(shape, array, dtype=array.dtype)
        else:
            try:
                array = np.broadcast_to(array, shape)
            except ValueError as error:
                raise ValueError(
                    f"{description} must match sparse indices"
                ) from error
        if not (
            array.dtype == np.dtype(np.bool_)
            or np.issubdtype(array.dtype, np.integer)
        ):
            raise TypeError(f"{description} must be integers")
        return array

    # Short alias for callers that already carry a coordinate tuple.
    def memberships(self, coordinate) -> tuple[int, ...]:
        try:
            row, column = coordinate
        except (TypeError, ValueError) as error:
            raise ValueError("Coordinate must contain row and column") from error
        return self.memberships_at(row, column)

    def count_class(self, value: int) -> int:
        """Return the exact number of pixels belonging to one class."""

        index = self._require_class(value)
        return int(
            _BYTE_POPCOUNT[self._packed_masks[index]].sum(dtype=np.uint64)
        )

    def _project_by_global_order(self, dtype) -> np.ndarray:
        """Build an initial projection when no per-pixel top is supplied."""

        result = np.zeros(self._shape, dtype=dtype)
        height, width = self._shape
        rows_per_chunk = max(
            1,
            min(
                height or 1,
                _SCAN_BYTES // max(1, width * result.dtype.itemsize),
            ),
        )
        for row_start in range(0, height, rows_per_chunk):
            row_stop = min(height, row_start + rows_per_chunk)
            chunk = result[row_start:row_stop]
            for value in self._z_order:
                membership = self._plane_patch(
                    value,
                    (row_start, 0),
                    (row_stop - row_start, width),
                )
                chunk[membership] = value
        return result

    def _projection_memberships_valid(self, projection=None) -> bool:
        """Check top membership and exact background semantics in chunks."""

        source = self._projection if projection is None else np.asarray(projection)
        height, width = self._shape
        rows_per_chunk = max(
            1,
            min(height or 1, _SCAN_BYTES // max(1, width * 3)),
        )
        for row_start in range(0, height, rows_per_chunk):
            row_stop = min(height, row_start + rows_per_chunk)
            projection_chunk = source[row_start:row_stop]
            occupied = np.zeros(projection_chunk.shape, dtype=bool)
            valid_top = np.zeros(projection_chunk.shape, dtype=bool)
            for value in self._class_values:
                membership = self._plane_patch(
                    value,
                    (row_start, 0),
                    (row_stop - row_start, width),
                )
                occupied |= membership
                valid_top |= membership & (projection_chunk == value)
            if np.any((projection_chunk == 0) != ~occupied):
                return False
            if np.any((projection_chunk != 0) & ~valid_top):
                return False
        return True

    def _reveal_patch(
        self,
        projection_patch: np.ndarray,
        reveal: np.ndarray,
        offset,
        *,
        exclude: int,
    ) -> None:
        """Reveal stable-z fallback memberships only at selected pixels."""

        projection_patch[reveal] = 0
        for value in self._z_order:
            if value == int(exclude):
                continue
            membership = self._plane_patch(value, offset, reveal.shape)
            projection_patch[reveal & membership] = value

    def project(
        self,
        dtype=None,
        *,
        exclude=None,
    ) -> np.ndarray:
        """Return a conventional topmost-class 2-D compatibility mask."""

        result_dtype = _safe_projection_dtype(
            self._projection_dtype if dtype is None else dtype,
            self._class_values,
        )
        excluded = _excluded_values(exclude)
        if not excluded:
            return np.array(
                self._projection,
                dtype=result_dtype,
                copy=True,
                order="C",
            )
        result = np.zeros(self._shape, dtype=result_dtype)
        height, width = self._shape
        rows_per_chunk = max(
            1,
            min(
                height or 1,
                _SCAN_BYTES
                // max(1, width * max(1, result_dtype.itemsize)),
            ),
        )
        for row_start in range(0, height, rows_per_chunk):
            row_stop = min(height, row_start + rows_per_chunk)
            result[row_start:row_stop] = self.project_patch(
                (row_start, 0),
                (row_stop - row_start, width),
                dtype=result_dtype,
                exclude=exclude,
            )
        return result

    def project_patch(
        self,
        offset,
        shape,
        dtype=None,
        *,
        exclude=None,
    ) -> np.ndarray:
        """Project a rectangle without unpacking any complete class plane."""

        try:
            patch_shape = tuple(int(size) for size in shape)
        except TypeError as error:
            raise ValueError(
                "Projection patch shape must have two dimensions"
            ) from error
        if len(patch_shape) != 2 or any(size < 0 for size in patch_shape):
            raise ValueError(
                "Projection patch shape must have two nonnegative dimensions"
            )
        self._validate_patch_bounds(offset, patch_shape)
        result_dtype = _safe_projection_dtype(
            self._projection_dtype if dtype is None else dtype,
            self._class_values,
        )
        row, column, row_stop, column_stop = self._validate_patch_bounds(
            offset,
            patch_shape,
        )
        excluded = _excluded_values(exclude)
        if not excluded:
            return np.array(
                self._projection[row:row_stop, column:column_stop],
                dtype=result_dtype,
                copy=True,
                order="C",
            )
        result = np.array(
            self._projection[row:row_stop, column:column_stop],
            dtype=result_dtype,
            copy=True,
            order="C",
        )
        needs_reveal = np.isin(result, tuple(excluded))
        if not np.any(needs_reveal):
            return result
        result[needs_reveal] = 0
        for value in self._z_order:
            if value in excluded:
                continue
            membership = self._plane_patch(value, offset, patch_shape)
            result[needs_reveal & membership] = value
        return result

    def add_class(
        self,
        value: int,
        name: str,
        color=None,
        *,
        top: bool = True,
    ) -> None:
        """Add one empty class plane and its lossless export metadata."""

        value = int(value)
        if value <= 0:
            raise ValueError("New class values must be positive")
        if value in self._class_map:
            raise ValueError(f"Class value {value} already exists")
        name = str(name).strip()
        if not name:
            raise ValueError("Class name cannot be blank")
        normalized_color = _normalized_metadata_color(
            color,
            f"Color for class {value}",
        )

        old_values = self._class_values
        new_values = tuple(sorted((*old_values, value)))
        new_projection_dtype = _safe_projection_dtype(
            self._projection_dtype,
            new_values,
        )
        height, packed_width = self._shape[0], (self._shape[1] + 7) // 8
        new_packed = np.zeros(
            (len(new_values), height, packed_width),
            dtype=np.uint8,
        )
        old_lookup = {old_value: i for i, old_value in enumerate(old_values)}
        for new_index, class_value in enumerate(new_values):
            if class_value in old_lookup:
                new_packed[new_index] = self._packed_masks[
                    old_lookup[class_value]
                ]
        new_projection = self._projection
        if new_projection_dtype != self._projection.dtype:
            new_projection = _copy_projection(
                self._projection,
                new_projection_dtype,
            )
        new_class_map = dict(self._class_map)
        new_class_map[value] = name
        new_class_colors = dict(self._class_colors)
        if normalized_color:
            new_class_colors[value] = normalized_color
        new_value_to_index = {
            class_value: index
            for index, class_value in enumerate(new_values)
        }
        new_z_order = (
            (*self._z_order, value) if top else (value, *self._z_order)
        )

        # Commit only after every potentially failing allocation completed.
        self._class_map = new_class_map
        self._class_colors = new_class_colors
        self._class_values = new_values
        self._value_to_index = new_value_to_index
        self._packed_masks = new_packed
        self._z_order = new_z_order
        self._projection = new_projection
        self._projection_dtype = new_projection_dtype
        self._mark_modified()

    def set_class_metadata(
        self,
        value: int,
        *,
        name: Optional[str] = None,
        color=None,
    ) -> None:
        """Update a class name/color without changing any membership bits."""

        value = int(value)
        if value not in self._class_map:
            raise ValueError(f"Unknown class value {value}")
        if name is not None:
            normalized_name = str(name).strip()
            if not normalized_name:
                raise ValueError("Class name cannot be blank")
            self._class_map[value] = normalized_name
        if color is not None:
            normalized_color = _normalized_metadata_color(
                color,
                f"Color for class {value}",
            )
            if normalized_color:
                self._class_colors[value] = normalized_color
            else:
                self._class_colors.pop(value, None)
        self._mark_modified()

    def remove_class(
        self,
        value: int,
        replacement: Optional[int] = None,
    ) -> int:
        """Remove a class, optionally OR-reassigning membership to another.

        ``replacement=0`` is equivalent to no replacement because background
        is represented by the absence of all foreground memberships.
        """

        value = int(value)
        source_index = self._require_class(value)
        replacement_index = None
        if replacement is not None:
            replacement = int(replacement)
            if replacement == value:
                raise ValueError("A class cannot be replaced by itself")
            if replacement < 0:
                raise ValueError("Replacement class cannot be negative")
            if replacement != 0:
                replacement_index = self._require_class(replacement)

        source_count = int(
            _BYTE_POPCOUNT[self._packed_masks[source_index]].sum(
                dtype=np.uint64
            )
        )
        new_values = tuple(
            class_value
            for class_value in self._class_values
            if class_value != value
        )
        new_value_to_index = {
            class_value: index
            for index, class_value in enumerate(new_values)
        }
        # np.delete is the largest allocation in class removal. Build the
        # complete candidate first so MemoryError leaves the live store exact.
        new_packed = np.delete(
            self._packed_masks,
            source_index,
            axis=0,
        )
        if replacement_index is not None:
            new_replacement_index = new_value_to_index[int(replacement)]
            np.bitwise_or(
                new_packed[new_replacement_index],
                self._packed_masks[source_index],
                out=new_packed[new_replacement_index],
            )
        new_projection = _copy_projection(
            self._projection,
            self._projection_dtype,
        )

        # Only pixels where the deleted class is currently visible can change
        # the compatibility projection. Hidden source memberships reassigned
        # into another class remain hidden, preserving unrelated top labels.
        height, width = self._shape
        rows_per_chunk = max(
            1,
            min(height or 1, _SCAN_BYTES // max(1, width * 2)),
        )
        for row_start in range(0, height, rows_per_chunk):
            row_stop = min(height, row_start + rows_per_chunk)
            projection_chunk = new_projection[row_start:row_stop]
            visible_source = projection_chunk == value
            if not np.any(visible_source):
                continue
            if replacement not in {None, 0}:
                projection_chunk[visible_source] = replacement
            else:
                self._reveal_patch(
                    projection_chunk,
                    visible_source,
                    (row_start, 0),
                    exclude=value,
                )
        new_class_map = dict(self._class_map)
        new_class_map.pop(value)
        new_class_colors = dict(self._class_colors)
        new_class_colors.pop(value, None)
        new_z_order = tuple(
            class_value for class_value in self._z_order if class_value != value
        )

        # All candidate state is complete; these reference swaps cannot leave
        # a half-removed class if an earlier allocation failed.
        self._packed_masks = new_packed
        self._projection = new_projection
        self._class_map = new_class_map
        self._class_colors = new_class_colors
        self._class_values = new_values
        self._value_to_index = new_value_to_index
        self._z_order = new_z_order
        self._mark_modified()
        return source_count

    def _mark_modified(self) -> None:
        self._projection_sha256 = ""
        self._generation = ""
        self._revision += 1

    def _projection_matches(self, projection) -> bool:
        array = _validate_label_array(projection, "Projection")
        if tuple(array.shape) != self._shape:
            return False
        rows_per_chunk = max(
            1,
            min(
                self._shape[0] or 1,
                _SCAN_BYTES // max(1, self._shape[1] * array.dtype.itemsize),
            ),
        )
        for row_start in range(0, self._shape[0], rows_per_chunk):
            row_stop = min(self._shape[0], row_start + rows_per_chunk)
            if not np.array_equal(
                array[row_start:row_stop],
                self._projection[row_start:row_stop],
            ):
                return False
        return True

    def to_payload(self, projection=None) -> bytes:
        """Serialize every annotation and class definition to NPZ bytes.

        The caller embeds these returned bytes inside the exact TIFF/PNG being
        saved.  The supplied 2-D projection must be the store's current
        topmost-class projection; this prevents a payload/image mismatch from
        being created accidentally.
        """

        if projection is None:
            projection_array = self.project()
        else:
            projection_array = _validate_label_array(projection, "Projection")
        if tuple(projection_array.shape) != self._shape:
            raise ValueError(
                f"Projection shape {projection_array.shape} does not match "
                f"overlap shape {self._shape}"
            )
        if not self._projection_matches(projection_array):
            raise ValueError(
                "Projection does not match the overlap store's visible top labels"
            )

        projection_dtype = _canonical_projection_dtype(projection_array.dtype)
        projection_digest = hash_projection(projection_array)
        generation = uuid.uuid4().hex
        names = _string_array(
            self._class_map[value] for value in self._class_values
        )
        colors = _string_array(
            self._class_colors.get(value, "") for value in self._class_values
        )
        background_name = str(self._class_map[0])
        background_color = self._class_colors.get(0, "")

        stream = io.BytesIO()
        np.savez_compressed(
            stream,
            format_version=np.asarray(FORMAT_VERSION, dtype=np.uint16),
            shape=np.asarray(self._shape, dtype=np.int64),
            class_values=np.asarray(self._class_values, dtype=np.int64),
            class_names=names,
            class_colors=colors,
            background_name=np.asarray(background_name),
            background_color=np.asarray(background_color),
            packed_masks=self._packed_masks,
            z_order=np.asarray(self._z_order, dtype=np.int64),
            projection_dtype=np.asarray(projection_dtype.str),
            projection_sha256=np.asarray(projection_digest),
            generation=np.asarray(generation),
        )
        return stream.getvalue()

    @classmethod
    def from_payload(cls, payload, projection=None) -> "OverlapStore":
        """Strictly deserialize an embedded lossless annotation payload."""

        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("Annotation payload must be bytes-like")
        try:
            with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as archive:
                keys = frozenset(archive.files)
                if keys != _REQUIRED_PAYLOAD_KEYS:
                    missing = sorted(_REQUIRED_PAYLOAD_KEYS - keys)
                    extra = sorted(keys - _REQUIRED_PAYLOAD_KEYS)
                    raise ValueError(
                        f"Invalid annotation payload keys; missing={missing}, "
                        f"extra={extra}"
                    )
                # NpzFile returns independent, owning ndarrays that remain
                # valid after the archive closes. Copying them again doubled
                # the peak for a real ~450 MB packed mask for no safety gain.
                arrays = {key: archive[key] for key in keys}
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("Annotation payload is not a valid NPZ archive") from error

        version = arrays["format_version"]
        if version.shape != () or version.dtype != np.dtype(np.uint16):
            raise ValueError("format_version must be a scalar uint16")
        if int(version) != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported annotation payload version {int(version)}"
            )

        shape_array = arrays["shape"]
        if shape_array.dtype != np.dtype(np.int64) or shape_array.shape != (2,):
            raise ValueError("shape must be an int64 array with two elements")
        shape = tuple(int(size) for size in shape_array)
        if any(size < 0 for size in shape):
            raise ValueError("Annotation shape cannot contain negative values")

        class_values_array = arrays["class_values"]
        z_order_array = arrays["z_order"]
        if (
            class_values_array.dtype != np.dtype(np.int64)
            or class_values_array.ndim != 1
        ):
            raise ValueError("class_values must be a one-dimensional int64 array")
        if z_order_array.dtype != np.dtype(np.int64) or z_order_array.ndim != 1:
            raise ValueError("z_order must be a one-dimensional int64 array")
        class_values = tuple(int(value) for value in class_values_array)
        if class_values != tuple(sorted(class_values)):
            raise ValueError("class_values must be strictly sorted")
        if any(value <= 0 for value in class_values) or len(set(class_values)) != len(
            class_values
        ):
            raise ValueError("class_values must contain unique positive integers")

        class_names_array = arrays["class_names"]
        class_colors_array = arrays["class_colors"]
        if (
            class_names_array.ndim != 1
            or class_names_array.dtype.kind != "U"
            or class_names_array.shape != class_values_array.shape
        ):
            raise ValueError("class_names must be a Unicode array aligned to classes")
        if (
            class_colors_array.ndim != 1
            or class_colors_array.dtype.kind != "U"
            or class_colors_array.shape != class_values_array.shape
        ):
            raise ValueError("class_colors must be a Unicode array aligned to classes")

        def scalar_unicode(key: str) -> str:
            value = arrays[key]
            if value.shape != () or value.dtype.kind != "U":
                raise ValueError(f"{key} must be a Unicode scalar")
            return str(value.item())

        background_name = scalar_unicode("background_name").strip()
        if not background_name:
            raise ValueError("background_name cannot be blank")
        background_color = _strict_payload_color(
            scalar_unicode("background_color"),
            "background_color",
        )
        names = tuple(str(name).strip() for name in class_names_array)
        if any(not name for name in names):
            raise ValueError("class_names cannot contain blank names")
        colors = tuple(
            _strict_payload_color(str(color), f"Color for class {value}")
            for value, color in zip(class_values, class_colors_array)
        )

        packed_masks = arrays["packed_masks"]
        if packed_masks.dtype != np.dtype(np.uint8) or packed_masks.ndim != 3:
            raise ValueError("packed_masks must be a three-dimensional uint8 array")
        expected_packed_shape = (
            len(class_values),
            shape[0],
            (shape[1] + 7) // 8,
        )
        if packed_masks.shape != expected_packed_shape:
            raise ValueError(
                f"packed_masks must have shape {expected_packed_shape}. Got "
                f"{packed_masks.shape}"
            )

        projection_dtype_text = scalar_unicode("projection_dtype")
        try:
            projection_dtype = _canonical_projection_dtype(
                np.dtype(projection_dtype_text)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("projection_dtype is invalid") from error
        if projection_dtype.str != projection_dtype_text:
            raise ValueError("projection_dtype is not in canonical form")

        projection_digest = scalar_unicode("projection_sha256")
        if re.fullmatch(r"[0-9a-f]{64}", projection_digest) is None:
            raise ValueError(
                "projection_sha256 must contain 64 lowercase hex digits"
            )
        generation = scalar_unicode("generation")
        if _GENERATION.fullmatch(generation) is None:
            raise ValueError("generation must contain 32 lowercase hex digits")

        class_map = {0: background_name}
        class_map.update(zip(class_values, names))
        class_colors: dict[int, str] = {}
        if background_color:
            class_colors[0] = background_color
        class_colors.update(
            (value, color)
            for value, color in zip(class_values, colors)
            if color
        )
        if projection is None:
            projection_array = None
        else:
            projection_array = _validate_label_array(projection, "Projection")
            if tuple(projection_array.shape) != shape:
                raise ValueError(
                    f"Projection shape {projection_array.shape} does not match "
                    f"payload shape {shape}"
                )
            if _canonical_projection_dtype(projection_array.dtype) != projection_dtype:
                raise ValueError(
                    "Projection dtype does not match the embedded annotation payload"
                )
        store = cls(
            shape,
            class_map,
            packed_masks,
            class_colors=class_colors,
            z_order=tuple(int(value) for value in z_order_array),
            projection_dtype=projection_dtype,
            projection_sha256=projection_digest,
            generation=generation,
            projection=projection_array,
        )
        if projection_array is None:
            projection_array = store.project(dtype=projection_dtype)
        if hash_projection(projection_array) != projection_digest:
            raise ValueError(
                "Projection hash does not match the embedded annotation payload"
            )
        return store
