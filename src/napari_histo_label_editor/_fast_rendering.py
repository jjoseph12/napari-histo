"""Responsive rendering for oversized editable napari Labels layers.

napari 0.6 downsamples a Labels texture when either displayed axis exceeds
``GL_MAX_TEXTURE_SIZE``.  Its partial-update handler then sees that the GPU
texture and raw label array have different shapes and falls back to a complete
layer refresh after every brush update.  This module maps each small changed
rectangle onto the already-downsampled texture instead.

The stock polygon overlay also triangulates a translucent filled polygon on
every mouse movement. The optimized preview uses a crisp outline and compact
high-contrast vertices, avoiding that repeated triangulation.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np


POLYGON_LINE_WIDTH = 2.0
POLYGON_VERTEX_SIZE = 8.0
POLYGON_VERTEX_EDGE_WIDTH = 1.25


def _qt_viewer(viewer: Any) -> Any | None:
    window = getattr(viewer, "window", None)
    return getattr(window, "_qt_viewer", None)


def _labels_visual(viewer: Any, layer: Any) -> Any | None:
    qt_viewer = _qt_viewer(viewer)
    mapping = getattr(qt_viewer, "layer_to_visual", None)
    if mapping is None:
        canvas = getattr(qt_viewer, "canvas", None)
        mapping = getattr(canvas, "layer_to_visual", None)
    if mapping is None:
        return None
    try:
        return mapping[layer]
    except (KeyError, TypeError):
        return None


def fast_partial_labels_update(visual: Any, event: Any) -> None:
    """Upload only the changed part of a downsampled Labels texture."""
    layer = visual.layer
    if not layer.loaded:
        return

    original = getattr(
        visual,
        "_napari_histo_original_partial_labels_update",
        None,
    )

    def fall_back() -> None:
        if original is not None:
            original(event)
        else:
            layer.refresh()

    try:
        offsets = np.asarray(event.offset, dtype=np.intp)
        update = np.asarray(event.data)
        dimensions = len(offsets)
        raw_shape = np.asarray(
            layer._slice.image.raw.shape[:dimensions],
            dtype=np.intp,
        )
        texture = visual.node._texture
        texture_shape = np.asarray(
            texture.shape[:dimensions],
            dtype=np.intp,
        )
    except (AttributeError, TypeError, ValueError):
        fall_back()
        return

    if (
        dimensions == 0
        or update.ndim < dimensions
        or len(raw_shape) != dimensions
        or len(texture_shape) != dimensions
        or np.any(raw_shape < 1)
        or np.any(texture_shape < 1)
    ):
        fall_back()
        return

    # napari downsamples as data[::factor]. Infer that integer factor from
    # the texture shape, then validate the result before touching the GPU.
    factors = np.ceil(raw_shape / texture_shape).astype(np.intp)
    expected_texture_shape = (raw_shape + factors - 1) // factors
    if not np.array_equal(expected_texture_shape, texture_shape):
        fall_back()
        return

    slices = []
    texture_offset = []
    for axis, (offset, factor) in enumerate(
        zip(offsets, factors, strict=True)
    ):
        first_sample = int((-offset) % factor)
        if first_sample >= update.shape[axis]:
            # The changed raw pixels fall between samples in the displayed
            # texture, so there is nothing to upload for this event.
            return
        slices.append(slice(first_sample, None, int(factor)))
        texture_offset.append(int((offset + first_sample) // factor))

    slices.extend([slice(None)] * (update.ndim - dimensions))
    sampled_update = update[tuple(slices)]
    sampled_shape = np.asarray(
        sampled_update.shape[:dimensions],
        dtype=np.intp,
    )
    texture_offset_array = np.asarray(texture_offset, dtype=np.intp)
    if (
        sampled_update.size == 0
        or np.any(texture_offset_array < 0)
        or np.any(texture_offset_array + sampled_shape > texture_shape)
    ):
        fall_back()
        return

    texture.scale_and_set_data(
        sampled_update,
        copy=False,
        offset=tuple(texture_offset),
    )
    visual.node.update()


def enable_fast_texture_updates(viewer: Any, layer: Any) -> bool:
    """Install downsample-aware partial GPU updates on a Labels visual."""
    visual = _labels_visual(viewer, layer)
    if visual is None:
        return False

    current = getattr(visual, "_on_partial_labels_update", None)
    if getattr(current, "__func__", None) is fast_partial_labels_update:
        return True
    if current is None or not hasattr(layer.events, "labels_update"):
        return False

    emitter = layer.events.labels_update
    try:
        emitter.disconnect(current)
    except (RuntimeError, ValueError):
        return False

    replacement = MethodType(fast_partial_labels_update, visual)
    try:
        visual._napari_histo_original_partial_labels_update = current
        visual._on_partial_labels_update = replacement
        emitter.connect(replacement)
    except Exception:
        visual._on_partial_labels_update = current
        emitter.connect(current)
        return False
    return True


def fast_polygon_points_change(visual: Any, event: Any = None) -> None:
    """Draw an inexpensive, high-visibility outline while making a polygon."""
    del event
    number_of_points = len(visual.overlay.points)
    if number_of_points:
        displayed_axes = np.asarray(visual._dims_displayed[::-1])
        points = np.asarray(visual.overlay.points, dtype=float)[
            :, displayed_axes
        ]

        # The overlay is parented to the Labels visual and therefore inherits
        # napari's tile2data scale.  Mouse points are already full-resolution
        # data coordinates, so compensate here or an oversized texture's
        # polygon preview is displaced (for example, 2x along a wide axis).
        try:
            tile_scale = np.asarray(
                visual.layer._transforms["tile2data"].scale,
                dtype=float,
            )[displayed_axes]
        except (AttributeError, KeyError, TypeError, ValueError):
            tile_scale = np.ones(points.shape[1], dtype=float)
        if (
            tile_scale.shape == (points.shape[1],)
            and np.all(np.isfinite(tile_scale))
            and np.all(tile_scale > 0)
        ):
            # Children receive napari's +0.5 pixel-center offset before their
            # parent scale and -0.5 master offset. Invert that complete mapping
            # so the outline lands exactly under the cursor, including at 2x.
            points = (points + 0.5) / tile_scale - 0.5
    else:
        points = np.empty((0, 2))

    # Never construct the filled preview: VisPy triangulates it on every mouse
    # move. A closed outline plus compact vertices is faster and easy to see.
    visual._polygon.visible = False
    visual._line.visible = number_of_points >= 2
    if visual._line.visible:
        outline = points
        if number_of_points > 2:
            outline = np.concatenate([points, points[:1]], axis=0)
        visual._line.set_data(pos=outline, width=POLYGON_LINE_WIDTH)

    visual._nodes.set_data(pos=points, **visual._nodes_kwargs)


def optimize_polygon_preview(viewer: Any, layer: Any) -> bool:
    """Install the fast, compact outline-and-vertices polygon preview."""
    qt_viewer = _qt_viewer(viewer)
    canvas = getattr(qt_viewer, "canvas", None)
    overlay_mapping = getattr(canvas, "_layer_overlay_to_visual", None)
    if overlay_mapping is None:
        return False

    try:
        overlay_model = layer._overlays["polygon"]
        visual = overlay_mapping[layer][overlay_model]
    except (AttributeError, KeyError, TypeError):
        return False

    current = getattr(visual, "_on_points_change", None)
    if getattr(current, "__func__", None) is fast_polygon_points_change:
        return True
    if current is None:
        return False

    emitter = visual.overlay.events.points
    try:
        emitter.disconnect(current)
    except (RuntimeError, ValueError):
        return False

    replacement = MethodType(fast_polygon_points_change, visual)
    replacement_connected = False
    try:
        # Keep VisPy's existing line implementation. Changing ``method`` after
        # the overlay is attached replaces an internal subvisual without its
        # inherited clip filter; napari then crashes when it closes/recreates
        # overlays after another layer is added.
        visual._line.set_data(width=POLYGON_LINE_WIDTH)
        visual._nodes_kwargs.update(
            size=POLYGON_VERTEX_SIZE,
            edge_width=POLYGON_VERTEX_EDGE_WIDTH,
            face_color=(1.0, 1.0, 1.0, 0.9),
            edge_color=(0.1, 0.1, 0.1, 1.0),
        )
        visual._napari_histo_original_polygon_points_change = current
        visual._on_points_change = replacement
        emitter.connect(replacement)
        replacement_connected = True
        replacement()
    except Exception:
        if replacement_connected:
            emitter.disconnect(replacement)
        visual._on_points_change = current
        emitter.connect(current)
        return False
    return True


def enable_fast_rendering(viewer: Any, layer: Any) -> tuple[bool, bool]:
    """Enable oversized-texture and polygon-preview optimizations."""
    return (
        enable_fast_texture_updates(viewer, layer),
        optimize_polygon_preview(viewer, layer),
    )


__all__ = [
    "enable_fast_rendering",
    "enable_fast_texture_updates",
    "fast_partial_labels_update",
    "fast_polygon_points_change",
    "optimize_polygon_preview",
]
