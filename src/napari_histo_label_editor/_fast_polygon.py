"""Bounding-box polygon painting for large napari Labels layers."""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING

import numpy as np
from skimage.draw import polygon2mask

if TYPE_CHECKING:
    from napari.layers import Labels


def fast_paint_polygon(layer: "Labels", points, new_label: int) -> None:
    """Paint a polygon while allocating a mask only for its bounding box.

    Napari 0.6 constructs ``polygon2mask`` with the complete label image
    shape.  A small polygon on a slide-sized mask therefore allocates and
    scans hundreds of millions of boolean pixels.  This equivalent path clips
    the polygon's bounding rectangle to the image and offsets the resulting
    indices before delegating to napari's normal paint/history machinery.
    """
    shape, dims_to_paint = layer._get_shape_and_dims_to_paint()
    if len(dims_to_paint) != 2:
        raise NotImplementedError(
            "Polygon painting is implemented only in 2D."
        )

    points = np.asarray(points, dtype=int)
    if points.ndim != 2 or len(points) == 0:
        return

    slice_coord = points[0].tolist()
    points_2d = points[:, dims_to_paint]
    paint_shape = np.asarray(shape, dtype=int)

    lower = np.maximum(points_2d.min(axis=0), 0)
    upper = np.minimum(points_2d.max(axis=0), paint_shape - 1)
    if np.any(upper < lower):
        return

    local_shape = tuple((upper - lower + 1).tolist())
    local_points = points_2d - lower
    local_mask = polygon2mask(local_shape, local_points)
    mask_indices = np.argwhere(local_mask)
    mask_indices += lower

    layer._paint_indices(
        mask_indices,
        new_label,
        shape,
        dims_to_paint,
        slice_coord,
        refresh=True,
    )


def enable_fast_polygon(layer: "Labels") -> None:
    """Install :func:`fast_paint_polygon` on one layer, idempotently."""
    current = getattr(layer, "paint_polygon", None)
    if getattr(current, "__func__", None) is fast_paint_polygon:
        return
    layer.paint_polygon = MethodType(fast_paint_polygon, layer)
