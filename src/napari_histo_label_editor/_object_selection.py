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


MAX_SELECTION_PIXELS = 500_000
MAX_EXACT_OUTLINE_PIXELS = 16 * 1024 * 1024
MAX_OUTLINE_VERTICES = 1024


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

    @property
    def outline(self) -> np.ndarray:
        """Return the largest display loop for backward-compatible callers."""

        if not self.outlines:
            return np.empty((0, 2), dtype=float)
        return self.outlines[0]

    @property
    def pixel_count(self) -> int:
        return int(self.rows.size)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        if not self.pixel_count:
            raise ValueError("An annotation selection cannot be empty")
        return (
            int(self.rows.min()),
            int(self.rows.max()) + 1,
            int(self.columns.min()),
            int(self.columns.max()) + 1,
        )

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
        height, width = (int(size) for size in shape)
        row_delta = int(row_delta)
        column_delta = int(column_delta)
        if self.rows.size:
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
        rows = self.rows + row_delta
        columns = self.columns + column_delta
        offset = np.array([row_delta, column_delta], dtype=float)
        return AnnotationObjectSelection(
            value=self.value,
            rows=np.ascontiguousarray(rows, dtype=np.intp),
            columns=np.ascontiguousarray(columns, dtype=np.intp),
            outlines=tuple(
                np.ascontiguousarray(outline + offset, dtype=float)
                for outline in self.outlines
            ),
            simplified_preview=self.simplified_preview,
            store_revision=self.store_revision,
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
    try:
        rows, columns = bounded_flood_indices(
            projection,
            (row, column),
            max_local_pixels=max_exact_outline_pixels,
            max_component_pixels=max_selection_pixels,
            allow_full_fallback=False,
        )
    except ValueError as error:
        if "connected component exceeds" in str(error):
            raise SelectionTooLargeError(
                "This visible region is too large to select safely. Use "
                "Erase/Fill for it, or select a smaller disconnected region."
            ) from error
        raise
    rows = np.ascontiguousarray(rows, dtype=np.intp)
    columns = np.ascontiguousarray(columns, dtype=np.intp)
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
    )


__all__ = [
    "AnnotationObjectSelection",
    "MAX_EXACT_OUTLINE_PIXELS",
    "MAX_OUTLINE_VERTICES",
    "MAX_SELECTION_PIXELS",
    "SelectionTooLargeError",
    "select_visible_component",
]
