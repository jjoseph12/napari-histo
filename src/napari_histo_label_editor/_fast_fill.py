"""Memory-conscious fill support for large napari Labels layers.

Napari 0.6 uses ``scipy.ndimage.label`` to identify the connected component
under the cursor.  That creates a full-size ``int32`` component image in
addition to boolean masks.  On a slide-sized label image, the temporary
component image alone can approach a gigabyte.

This module keeps napari's public ``Labels.fill`` behaviour while starting
``skimage.segmentation.flood`` in a bounded window around the clicked region.
The window expands only across proven same-label connections and falls back to
the full compiled search for genuinely large regions. Ordinary objects avoid
both napari's full ``int32`` component image and a full-slide flood map.
"""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING

import numpy as np
from skimage.segmentation import flood

if TYPE_CHECKING:
    from napari.layers import Labels


LOCAL_FLOOD_WINDOW = 1024
MAX_LOCAL_FLOOD_PIXELS = 16 * 1024 * 1024


def _mask_indices_with_offset(
    mask: np.ndarray,
    row_offset: int = 0,
    column_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(mask)
    rows += row_offset
    columns += column_offset
    return rows, columns


def bounded_flood_indices(
    labels: np.ndarray,
    seed: tuple[int, int],
    *,
    initial_window: int = LOCAL_FLOOD_WINDOW,
    max_local_pixels: int = MAX_LOCAL_FLOOD_PIXELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a 4-connected component without starting at full-slide size.

    The search begins in a window around ``seed``. It expands only when a
    component pixel connects to an equal-valued pixel immediately outside the
    current window. This is exact: when no such boundary connection exists,
    the complete global component has been found.

    If the local search itself grows large, the function falls back to
    scikit-image's full-array flood. That keeps giant/background fills on the
    efficient compiled path while avoiding a slide-sized temporary boolean
    array for ordinary bounded regions.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("bounded_flood_indices requires a 2D label array")
    if initial_window < 1:
        raise ValueError("initial_window must be at least 1")
    if max_local_pixels < 1:
        raise ValueError("max_local_pixels must be at least 1")

    height, width = labels.shape
    seed_row, seed_column = (int(seed[0]), int(seed[1]))
    old_label = labels[seed_row, seed_column]

    window_height = min(int(initial_window), height)
    window_width = min(int(initial_window), width)
    row_start = max(
        0,
        min(seed_row - window_height // 2, height - window_height),
    )
    column_start = max(
        0,
        min(seed_column - window_width // 2, width - window_width),
    )
    row_stop = row_start + window_height
    column_stop = column_start + window_width

    while True:
        local_labels = labels[row_start:row_stop, column_start:column_stop]
        local_seed = (seed_row - row_start, seed_column - column_start)
        matches = flood(local_labels, local_seed, connectivity=1)

        expand_top = bool(
            row_start > 0
            and np.any(
                matches[0]
                & (labels[row_start - 1, column_start:column_stop] == old_label)
            )
        )
        expand_bottom = bool(
            row_stop < height
            and np.any(
                matches[-1]
                & (labels[row_stop, column_start:column_stop] == old_label)
            )
        )
        expand_left = bool(
            column_start > 0
            and np.any(
                matches[:, 0]
                & (labels[row_start:row_stop, column_start - 1] == old_label)
            )
        )
        expand_right = bool(
            column_stop < width
            and np.any(
                matches[:, -1]
                & (labels[row_start:row_stop, column_stop] == old_label)
            )
        )

        if not (
            expand_top or expand_bottom or expand_left or expand_right
        ):
            return _mask_indices_with_offset(
                matches,
                row_start,
                column_start,
            )

        current_height = row_stop - row_start
        current_width = column_stop - column_start
        next_row_start = (
            max(0, row_start - current_height)
            if expand_top
            else row_start
        )
        next_row_stop = (
            min(height, row_stop + current_height)
            if expand_bottom
            else row_stop
        )
        next_column_start = (
            max(0, column_start - current_width)
            if expand_left
            else column_start
        )
        next_column_stop = (
            min(width, column_stop + current_width)
            if expand_right
            else column_stop
        )
        next_pixels = (next_row_stop - next_row_start) * (
            next_column_stop - next_column_start
        )

        if next_pixels > max_local_pixels:
            # Release the local mask before allocating the full fallback.
            del matches
            return np.nonzero(flood(labels, seed, connectivity=1))

        row_start, row_stop = next_row_start, next_row_stop
        column_start, column_stop = next_column_start, next_column_stop


def _visible_overlap_patch(
    composite: np.ndarray,
    active_mask: np.ndarray,
    active_value: int,
    row_slice,
    column_slice,
) -> np.ndarray:
    del active_mask, active_value
    return np.array(
        composite[row_slice, column_slice],
        copy=True,
        order="C",
    )


def bounded_overlap_flood_indices(
    composite: np.ndarray,
    active_mask: np.ndarray,
    active_value: int,
    seed: tuple[int, int],
    *,
    initial_window: int = LOCAL_FLOOD_WINDOW,
    max_local_pixels: int = MAX_LOCAL_FLOOD_PIXELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Flood one component in the authoritative visible ``composite``.

    Only the expanding local rectangle is materialized for ordinary objects.
    This keeps bucket fill fast for slide-sized annotations while respecting
    visible class boundaries that are absent from the binary edit mask. The
    active mask is intentionally not drawn on top: it is a transparent tool
    layer and can contain membership hidden below another visible class.
    """
    composite = np.asarray(composite)
    active_mask = np.asarray(active_mask)
    if composite.ndim != 2 or active_mask.ndim != 2:
        raise ValueError("Overlap fill requires two 2-D arrays")
    if composite.shape != active_mask.shape:
        raise ValueError("Composite and active mask shapes must match")
    if initial_window < 1:
        raise ValueError("initial_window must be at least 1")
    if max_local_pixels < 1:
        raise ValueError("max_local_pixels must be at least 1")

    height, width = composite.shape
    seed_row, seed_column = (int(seed[0]), int(seed[1]))
    if not (0 <= seed_row < height and 0 <= seed_column < width):
        raise IndexError("Overlap fill seed is outside the annotation")
    active_value = int(active_value)
    old_label = composite[seed_row, seed_column]

    window_height = min(int(initial_window), height)
    window_width = min(int(initial_window), width)
    row_start = max(
        0,
        min(seed_row - window_height // 2, height - window_height),
    )
    column_start = max(
        0,
        min(seed_column - window_width // 2, width - window_width),
    )
    row_stop = row_start + window_height
    column_stop = column_start + window_width

    def visible_patch(rs, cs):
        return _visible_overlap_patch(
            composite,
            active_mask,
            active_value,
            rs,
            cs,
        )

    while True:
        local_visible = visible_patch(
            slice(row_start, row_stop),
            slice(column_start, column_stop),
        )
        local_seed = (seed_row - row_start, seed_column - column_start)
        matches = flood(local_visible, local_seed, connectivity=1)

        expand_top = bool(
            row_start > 0
            and np.any(
                matches[0]
                & (
                    visible_patch(
                        row_start - 1,
                        slice(column_start, column_stop),
                    )
                    == old_label
                )
            )
        )
        expand_bottom = bool(
            row_stop < height
            and np.any(
                matches[-1]
                & (
                    visible_patch(
                        row_stop,
                        slice(column_start, column_stop),
                    )
                    == old_label
                )
            )
        )
        expand_left = bool(
            column_start > 0
            and np.any(
                matches[:, 0]
                & (
                    visible_patch(
                        slice(row_start, row_stop),
                        column_start - 1,
                    )
                    == old_label
                )
            )
        )
        expand_right = bool(
            column_stop < width
            and np.any(
                matches[:, -1]
                & (
                    visible_patch(
                        slice(row_start, row_stop),
                        column_stop,
                    )
                    == old_label
                )
            )
        )

        if not (expand_top or expand_bottom or expand_left or expand_right):
            return _mask_indices_with_offset(matches, row_start, column_start)

        current_height = row_stop - row_start
        current_width = column_stop - column_start
        next_row_start = (
            max(0, row_start - current_height) if expand_top else row_start
        )
        next_row_stop = (
            min(height, row_stop + current_height) if expand_bottom else row_stop
        )
        next_column_start = (
            max(0, column_start - current_width)
            if expand_left
            else column_start
        )
        next_column_stop = (
            min(width, column_stop + current_width)
            if expand_right
            else column_stop
        )
        next_pixels = (next_row_stop - next_row_start) * (
            next_column_stop - next_column_start
        )
        if next_pixels > max_local_pixels:
            del matches, local_visible
            visible = visible_patch(slice(None), slice(None))
            return np.nonzero(flood(visible, seed, connectivity=1))

        row_start, row_stop = next_row_start, next_row_stop
        column_start, column_stop = next_column_start, next_column_stop


def overlap_fill(
    layer: "Labels",
    coord,
    new_label: int,
    refresh: bool = True,
) -> None:
    """Fill the clicked visible class into one binary active-class layer."""
    int_coord = tuple(np.round(coord).astype(int))
    data = np.asarray(layer.data)
    if data.ndim != 2:
        raise ValueError("Overlap fill is implemented only in 2D")
    if np.any(np.less(int_coord, 0)) or np.any(
        np.greater_equal(int_coord, data.shape)
    ):
        return
    new_label = int(new_label)
    if new_label not in {0, 1}:
        raise ValueError("The active overlap layer accepts only 0 or 1")

    # Erasing is deliberately limited to the connected active membership.
    # It must not use the lower composite, or it could remove disconnected
    # active objects that merely reveal the same lower class.
    if new_label == 0:
        if int(data[int_coord]) == 0:
            return
        fast_fill(layer, int_coord, 0, refresh)
        return

    composite = getattr(layer, "_napari_histo_overlap_composite", None)
    active_value = getattr(layer, "_napari_histo_active_value", None)
    if composite is None or active_value is None:
        raise RuntimeError("Overlap fill is not bound to an active class")
    # A 1 in the transparent edit mask may be hidden under another visible
    # class. That is not a completed fill: use the authoritative composite's
    # component so absent neighboring memberships are still painted. Exact
    # repaint of the already-present seed membership remains idempotent; the
    # store adapter promotes only actual 0 -> 1 additions locally.
    if (
        int(data[int_coord]) == 1
        and int(np.asarray(composite)[int_coord]) == int(active_value)
    ):
        return

    if layer.contiguous:
        indices = bounded_overlap_flood_indices(
            composite,
            data,
            active_value,
            int_coord,
        )
    else:
        visible = _visible_overlap_patch(
            np.asarray(composite),
            data,
            int(active_value),
            slice(None),
            slice(None),
        )
        target = visible[int_coord]
        indices = np.nonzero(visible == target)
    layer.data_setitem(indices, 1, refresh)


def enable_overlap_fill(
    layer: "Labels",
    composite: np.ndarray,
    active_value: int | None,
) -> None:
    """Bind visible-composite bucket semantics to a binary Labels layer."""
    layer._napari_histo_overlap_composite = composite
    layer._napari_histo_active_value = active_value
    layer.fill = MethodType(overlap_fill, layer)


def fast_fill(
    layer: "Labels",
    coord,
    new_label: int,
    refresh: bool = True,
) -> None:
    """Fill labels with lower peak memory than napari 0.6's implementation.

    The bounds, label-preservation, dimensional slicing, connectivity,
    history, and refresh behaviour intentionally mirror
    :meth:`napari.layers.Labels.fill` from napari 0.6.

    Parameters
    ----------
    layer : napari.layers.Labels
        Labels layer to modify.
    coord : sequence of float
        Cursor position in data coordinates.
    new_label : int
        Value to write into the filled region.
    refresh : bool, default True
        Forwarded to ``Labels.data_setitem`` so callers can batch refreshes.
    """
    int_coord = tuple(np.round(coord).astype(int))

    if np.any(np.less(int_coord, 0)) or np.any(
        np.greater_equal(int_coord, layer.data.shape)
    ):
        return

    old_label = np.asarray(layer.data[int_coord]).item()
    if old_label == new_label:
        return

    background_value = layer.colormap.background_value
    if (
        layer.preserve_labels
        and old_label != layer._prev_selected_label
        and old_label != background_value
    ):
        return

    dims_to_fill = sorted(
        layer._slice_input.order[-layer.n_edit_dimensions :]
    )
    data_slice_list = list(int_coord)
    for dim in dims_to_fill:
        data_slice_list[dim] = slice(None)
    data_slice = tuple(data_slice_list)
    labels = np.asarray(layer.data[data_slice])
    slice_coord = tuple(int_coord[dim] for dim in dims_to_fill)

    if layer.contiguous:
        # scipy.ndimage.label's default footprint uses orthogonal neighbors,
        # so connectivity=1 is required for exact napari equivalence.
        if labels.ndim == 2:
            match_indices_local = bounded_flood_indices(
                labels,
                slice_coord,
            )
            matches = None
        else:
            matches = flood(labels, slice_coord, connectivity=1)
            match_indices_local = np.nonzero(matches)
    else:
        matches = labels == old_label
        match_indices_local = np.nonzero(matches)

    if matches is not None:
        # ``nonzero`` owns its coordinate arrays, so release the mask before
        # ``data_setitem`` allocates its changed-pixel/history arrays.
        del matches
    if layer.ndim not in {2, layer.n_edit_dimensions}:
        n_indices = len(match_indices_local[0])
        match_indices = []
        local_dim = 0
        for index in data_slice:
            if isinstance(index, slice):
                match_indices.append(match_indices_local[local_dim])
                local_dim += 1
            else:
                match_indices.append(
                    np.full(n_indices, index, dtype=np.intp)
                )
        match_indices = tuple(match_indices)
    else:
        match_indices = match_indices_local

    layer.data_setitem(match_indices, new_label, refresh)


def enable_fast_fill(layer: "Labels") -> None:
    """Install :func:`fast_fill` on one Labels layer, idempotently."""
    current_fill = getattr(layer, "fill", None)
    if getattr(current_fill, "__func__", None) is fast_fill:
        return
    layer.fill = MethodType(fast_fill, layer)
