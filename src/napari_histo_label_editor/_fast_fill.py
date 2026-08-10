"""Memory-conscious fill support for large napari Labels layers.

Napari 0.6 uses ``scipy.ndimage.label`` to identify the connected component
under the cursor.  That creates a full-size ``int32`` component image in
addition to boolean masks.  On a slide-sized label image, the temporary
component image alone can approach a gigabyte.

This module keeps napari's public ``Labels.fill`` behaviour while using
``skimage.segmentation.flood`` for connected fills.  ``flood`` maintains a
byte-per-pixel visitation map instead of an ``int32`` component image.
"""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING

import numpy as np
from skimage.segmentation import flood

if TYPE_CHECKING:
    from napari.layers import Labels


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
        matches = flood(labels, slice_coord, connectivity=1)
    else:
        matches = labels == old_label

    match_indices_local = np.nonzero(matches)
    # ``nonzero`` owns its coordinate arrays, so release the full-size mask
    # before ``data_setitem`` allocates its changed-pixel/history arrays.
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
