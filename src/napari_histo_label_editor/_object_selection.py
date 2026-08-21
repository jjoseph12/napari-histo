"""Connected visible-annotation selection for the label editor.

Selections are intentionally sparse: the exact object is represented by row
and column coordinate arrays, while only a bounded local mask is created to
trace its display outline.  This avoids another slide-sized Labels layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from skimage.measure import approximate_polygon, find_contours

from ._fast_fill import bounded_flood_indices
from ._overlap_store import _UniqueRunIndices


# Ordinary components keep two compact coordinate arrays. Beyond sixteen
# million pixels, the picker switches to immutable row runs so broad tissue
# regions do not retain 8 bytes of coordinates per pixel.
MAX_SPARSE_SELECTION_PIXELS = 16 * 1024 * 1024

# Component discovery is allowed to inspect a larger bounded rectangle than
# the component itself occupies.  This is deliberately independent of the
# outline budget: a thin or perforated annotation can span most of a slide
# while still having a reasonably-sized exact sparse selection.  64 MiPixels
# covers the 7,048 x 8,001 annotation canvases this editor commonly opens and
# bounds the largest temporary flood mask to 64 MiB.
MAX_SELECTION_SEARCH_PIXELS = 64 * 1024 * 1024

# A pathological connected comb can have nearly one run per pixel. Bound the
# three signed run arrays independently even though dense real tissue usually
# needs only one or a few runs per image row.
MAX_SELECTION_RUN_BYTES = 64 * 1024 * 1024

# A component cannot contain more pixels than the bounded search rectangle.
# Keeping these limits equal lets every region on the editor's common
# 7,048 x 8,001 canvases reach the compact run encoder.
MAX_SELECTION_PIXELS = MAX_SELECTION_SEARCH_PIXELS

# ``find_contours`` uses float work arrays that are substantially larger than
# the uint8 local mask.  Above four MiPixels, retain the exact sparse object
# but show its explicitly-marked rectangular preview instead.
MAX_EXACT_OUTLINE_PIXELS = 4 * 1024 * 1024
MAX_OUTLINE_VERTICES = 1024


def _selection_index_dtype(shape: Sequence[int]) -> np.dtype:
    """Return the smallest signed index dtype that can address ``shape``."""

    largest_coordinate = max((int(size) - 1 for size in shape), default=0)
    if largest_coordinate <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


class SelectionTooLargeError(ValueError):
    """Raised before a selection would allocate unbounded slide memory."""


@dataclass(frozen=True)
class AnnotationObjectSelection:
    """One connected visible region in the semantic projection."""

    value: int
    rows: np.ndarray
    columns: np.ndarray
    outlines: tuple[np.ndarray, ...]
    simplified_preview: bool = False
    store_revision: int | None = None
    runs: _UniqueRunIndices | None = None

    @property
    def outline(self) -> np.ndarray:
        """Return the largest display loop for backward-compatible callers."""

        if not self.outlines:
            return np.empty((0, 2), dtype=float)
        return self.outlines[0]

    @property
    def pixel_count(self) -> int:
        if self.runs is not None:
            return self.runs.pixel_count
        return int(self.rows.size)

    @property
    def indices(
        self,
    ) -> tuple[np.ndarray, np.ndarray] | _UniqueRunIndices:
        """Return the trusted sparse or compact-run selection carrier."""

        if self.runs is not None:
            return self.runs
        return self.rows, self.columns

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        if not self.pixel_count:
            raise ValueError("An annotation selection cannot be empty")
        if self.runs is not None:
            return self.runs.bounds
        return (
            int(self.rows.min()),
            int(self.rows.max()) + 1,
            int(self.columns.min()),
            int(self.columns.max()) + 1,
        )

    def iter_indices(
        self,
        max_pixels: int = 256 * 1024,
    ):
        """Yield exact coordinate views in bounded read-only chunks.

        The views preserve the flood's unique row-major ordering and do not
        allocate new coordinate arrays. Selected-object Delete intentionally
        uses its dedicated atomic sparse path rather than this iterator.
        """

        max_pixels = int(max_pixels)
        if max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        if self.runs is not None:
            yield from self.runs.iter_chunks(max_pixels)
            return
        for start in range(0, self.pixel_count, max_pixels):
            stop = min(self.pixel_count, start + max_pixels)
            yield self.rows[start:stop], self.columns[start:stop]

    def translated(
        self,
        row_delta: int,
        column_delta: int,
        shape: Sequence[int],
    ) -> "AnnotationObjectSelection":
        """Return an exact integer translation inside ``shape``."""

        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError("Annotation movement requires a 2-D shape")
        if self.runs is not None:
            raise ValueError(
                "Moving a very large run-backed selection is not supported; "
                "Delete remains available."
            )
        height, width = (int(size) for size in shape)
        row_delta = int(row_delta)
        column_delta = int(column_delta)
        if self.pixel_count:
            row_start, row_stop, column_start, column_stop = self.bounds
            if (
                row_start + row_delta < 0
                or row_stop + row_delta > height
                or column_start + column_delta < 0
                or column_stop + column_delta > width
            ):
                raise ValueError(
                    "The selected annotation cannot move outside the image"
                )
        # Keep ordinary slide selections compact after a translation. The
        # bounds check above makes the same-width signed addition safe.
        row_offset = np.asarray(row_delta, dtype=self.rows.dtype)
        column_offset = np.asarray(column_delta, dtype=self.columns.dtype)
        rows = np.add(self.rows, row_offset, dtype=self.rows.dtype)
        columns = np.add(
            self.columns,
            column_offset,
            dtype=self.columns.dtype,
        )
        rows = np.ascontiguousarray(rows)
        columns = np.ascontiguousarray(columns)
        rows.setflags(write=False)
        columns.setflags(write=False)
        offset = np.array([row_delta, column_delta], dtype=float)
        return AnnotationObjectSelection(
            value=self.value,
            rows=rows,
            columns=columns,
            outlines=tuple(
                np.ascontiguousarray(outline + offset, dtype=float)
                for outline in self.outlines
            ),
            simplified_preview=self.simplified_preview,
            store_revision=self.store_revision,
            runs=None,
        )


def _bounding_outline(
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
) -> np.ndarray:
    """Return a pixel-edge rectangle in napari row/column coordinates."""

    top = row_start - 0.5
    bottom = row_stop - 0.5
    left = column_start - 0.5
    right = column_stop - 0.5
    return np.array(
        [[top, left], [top, right], [bottom, right], [bottom, left]],
        dtype=float,
    )


def _component_outlines(
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    max_exact_pixels: int,
    max_vertices: int,
) -> tuple[tuple[np.ndarray, ...], bool]:
    row_start = int(rows.min())
    row_stop = int(rows.max()) + 1
    column_start = int(columns.min())
    column_stop = int(columns.max()) + 1
    height = row_stop - row_start
    width = column_stop - column_start
    if height * width > max_exact_pixels:
        return (
            (
                _bounding_outline(
                    row_start,
                    row_stop,
                    column_start,
                    column_stop,
                ),
            ),
            False,
        )

    # Padding lets find_contours close regions that touch their local bounds.
    local = np.zeros((height + 2, width + 2), dtype=np.uint8)
    local[rows - row_start + 1, columns - column_start + 1] = 1
    contours = find_contours(local, level=0.5, fully_connected="low")
    if not contours:
        return (
            (
                _bounding_outline(
                    row_start,
                    row_stop,
                    column_start,
                    column_stop,
                ),
            ),
            False,
        )

    # Preserve every outer and hole boundary exactly when it fits the display
    # budget. Simplification affects only the preview, never selected pixels.
    result = []
    remaining = int(max_vertices)
    normalized_contours = []
    for contour in sorted(contours, key=len, reverse=True):
        contour[:, 0] += row_start - 1
        contour[:, 1] += column_start - 1
        if contour.shape[0] > 1 and np.array_equal(contour[0], contour[-1]):
            contour = contour[:-1]
        if contour.shape[0] >= 2:
            normalized_contours.append(contour)
    exact = sum(len(contour) for contour in normalized_contours) <= max_vertices
    for contour in normalized_contours:
        if not exact:
            contour = approximate_polygon(contour, tolerance=0.35)
        if contour.shape[0] < 2 or remaining < 2:
            exact = False
            continue
        if contour.shape[0] > remaining:
            exact = False
            chosen = np.linspace(
                0,
                contour.shape[0] - 1,
                num=remaining,
                dtype=np.intp,
            )
            contour = contour[chosen]
        result.append(np.ascontiguousarray(contour, dtype=float))
        remaining -= contour.shape[0]
        if remaining < 2:
            if len(result) < len(normalized_contours):
                exact = False
            break
    if not result:
        result.append(
            _bounding_outline(
                row_start,
                row_stop,
                column_start,
                column_stop,
            )
        )
    return tuple(result), exact


def select_visible_component(
    projection: np.ndarray,
    seed: Sequence[int],
    *,
    max_selection_pixels: int = MAX_SELECTION_PIXELS,
    max_search_pixels: int = MAX_SELECTION_SEARCH_PIXELS,
    max_sparse_pixels: int = MAX_SPARSE_SELECTION_PIXELS,
    max_run_bytes: int = MAX_SELECTION_RUN_BYTES,
    max_exact_outline_pixels: int = MAX_EXACT_OUTLINE_PIXELS,
    max_outline_vertices: int = MAX_OUTLINE_VERTICES,
    store_revision: int | None = None,
) -> Optional[AnnotationObjectSelection]:
    """Select the 4-connected visible object containing ``seed``.

    Background returns ``None``.  Selection follows the public top-class
    projection, so deleting it reveals rather than destroys hidden overlap
    memberships.
    """

    projection = np.asarray(projection)
    if projection.ndim != 2:
        raise ValueError("Annotation object selection requires a 2-D projection")
    if max_exact_outline_pixels < 1:
        raise ValueError("max_exact_outline_pixels must be positive")
    if max_selection_pixels < 1:
        raise ValueError("max_selection_pixels must be positive")
    if max_search_pixels < 1:
        raise ValueError("max_search_pixels must be positive")
    if max_sparse_pixels < 0:
        raise ValueError("max_sparse_pixels cannot be negative")
    if max_run_bytes < 1:
        raise ValueError("max_run_bytes must be positive")
    if max_outline_vertices < 4:
        raise ValueError("max_outline_vertices must be at least 4")
    seed = tuple(seed)
    if len(seed) != 2:
        raise ValueError("Selection seed must contain row and column")
    row, column = (int(coordinate) for coordinate in seed)
    if not (0 <= row < projection.shape[0] and 0 <= column < projection.shape[1]):
        return None
    value = int(projection[row, column])
    if value == 0:
        return None
    index_dtype = _selection_index_dtype(projection.shape)
    try:
        component_indices = bounded_flood_indices(
            projection,
            (row, column),
            max_local_pixels=max_search_pixels,
            max_component_pixels=max_selection_pixels,
            allow_full_fallback=False,
            index_dtype=index_dtype,
            run_threshold_pixels=max_sparse_pixels,
            max_run_bytes=max_run_bytes,
        )
    except ValueError as error:
        message = str(error)
        if "pixel limit" in message:
            raise SelectionTooLargeError(
                "This connected region is too large to select safely; it "
                "exceeds the "
                f"{int(max_selection_pixels):,}-pixel selection limit."
            ) from error
        if "bounded flood window" in message:
            raise SelectionTooLargeError(
                "This connected region is too large to search safely; it "
                "spans more than the "
                f"{int(max_search_pixels):,}-pixel search window."
            ) from error
        if "run byte limit" in message:
            raise SelectionTooLargeError(
                "This connected region is too fragmented to select safely; "
                "its compact run index exceeds the "
                f"{int(max_run_bytes):,}-byte selection limit."
            ) from error
        raise

    if len(component_indices) == 3:
        run_rows, run_starts, run_stops = component_indices
        runs = _UniqueRunIndices(
            np.ascontiguousarray(run_rows, dtype=index_dtype),
            np.ascontiguousarray(run_starts, dtype=index_dtype),
            np.ascontiguousarray(run_stops, dtype=index_dtype),
        )
        rows = np.empty(0, dtype=index_dtype)
        columns = np.empty(0, dtype=index_dtype)
        rows.setflags(write=False)
        columns.setflags(write=False)
        row_start, row_stop, column_start, column_stop = runs.bounds
        outlines = (
            _bounding_outline(
                row_start,
                row_stop,
                column_start,
                column_stop,
            ),
        )
        full_outline = False
    else:
        rows, columns = component_indices
        runs = None
        # np.nonzero (used by the compiled flood path) returns platform-sized
        # indices. A 32-bit signed coordinate is sufficient for normal slide
        # dimensions and halves persistent selection memory on 64-bit systems.
        rows = np.ascontiguousarray(rows, dtype=index_dtype)
        columns = np.ascontiguousarray(columns, dtype=index_dtype)
        # Delete history deliberately retains these compact arrays without a
        # second N-pixel copy. Freeze them once the flood is final so stale UI
        # references cannot redirect a later Undo/Redo target.
        rows.setflags(write=False)
        columns.setflags(write=False)
        outlines, full_outline = _component_outlines(
            rows,
            columns,
            max_exact_pixels=int(max_exact_outline_pixels),
            max_vertices=int(max_outline_vertices),
        )
    return AnnotationObjectSelection(
        value=value,
        rows=rows,
        columns=columns,
        outlines=outlines,
        simplified_preview=not full_outline,
        store_revision=(
            None if store_revision is None else int(store_revision)
        ),
        runs=runs,
    )


__all__ = [
    "AnnotationObjectSelection",
    "MAX_EXACT_OUTLINE_PIXELS",
    "MAX_OUTLINE_VERTICES",
    "MAX_SELECTION_RUN_BYTES",
    "MAX_SPARSE_SELECTION_PIXELS",
    "MAX_SELECTION_PIXELS",
    "MAX_SELECTION_SEARCH_PIXELS",
    "SelectionTooLargeError",
    "select_visible_component",
]
