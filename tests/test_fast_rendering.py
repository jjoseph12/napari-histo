import unittest
from types import SimpleNamespace

import numpy as np

from napari_histo_label_editor._fast_rendering import (
    enable_fast_texture_updates,
    fast_partial_labels_update,
    fast_polygon_points_change,
    optimize_polygon_preview,
)


class FakeEmitter:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def disconnect(self, callback):
        if callback not in self.callbacks:
            raise ValueError("callback is not connected")
        self.callbacks.remove(callback)

    def emit(self, **kwargs):
        event = SimpleNamespace(**kwargs)
        for callback in list(self.callbacks):
            callback(event)


class FakeTexture:
    def __init__(self, shape):
        self.shape = shape
        self.uploads = []

    def scale_and_set_data(self, data, copy, offset):
        self.uploads.append(
            {
                "data": np.asarray(data).copy(),
                "copy": copy,
                "offset": offset,
            }
        )


class FakeNode:
    def __init__(self, texture_shape):
        self._texture = FakeTexture(texture_shape)
        self.update_count = 0

    def update(self):
        self.update_count += 1


class FakeLayer:
    def __init__(self, raw_shape):
        self.loaded = True
        self._slice = SimpleNamespace(
            image=SimpleNamespace(raw=np.zeros(raw_shape, dtype=np.uint8))
        )
        self.events = SimpleNamespace(labels_update=FakeEmitter())
        self.refresh_count = 0
        self._overlays = {}
        self._transforms = {
            "tile2data": SimpleNamespace(scale=np.ones(len(raw_shape)))
        }

    def refresh(self):
        self.refresh_count += 1


class FakeLabelsVisual:
    def __init__(self, layer, texture_shape):
        self.layer = layer
        self.node = FakeNode(texture_shape)
        self.original_count = 0

    def _on_partial_labels_update(self, event):
        del event
        self.original_count += 1


def make_viewer(layer, labels_visual, polygon_visual=None):
    canvas = SimpleNamespace(
        layer_to_visual={layer: labels_visual},
        _layer_overlay_to_visual={},
    )
    if polygon_visual is not None:
        canvas._layer_overlay_to_visual = {
            layer: {layer._overlays["polygon"]: polygon_visual}
        }
    qt_viewer = SimpleNamespace(
        layer_to_visual={layer: labels_visual},
        canvas=canvas,
    )
    return SimpleNamespace(
        window=SimpleNamespace(_qt_viewer=qt_viewer)
    )


class FastPartialLabelsUpdateTest(unittest.TestCase):
    def setup_visual(self, raw_shape, texture_shape):
        layer = FakeLayer(raw_shape)
        visual = FakeLabelsVisual(layer, texture_shape)
        layer.events.labels_update.connect(
            visual._on_partial_labels_update
        )
        viewer = make_viewer(layer, visual)
        return viewer, layer, visual

    def test_oversized_texture_uploads_only_sampled_changed_pixels(self):
        viewer, layer, visual = self.setup_visual((6, 10), (6, 5))
        self.assertTrue(enable_fast_texture_updates(viewer, layer))

        update = np.arange(8, dtype=np.uint8).reshape(2, 4)
        layer.events.labels_update.emit(data=update, offset=(1, 3))

        self.assertEqual(visual.original_count, 0)
        self.assertEqual(len(visual.node._texture.uploads), 1)
        upload = visual.node._texture.uploads[0]
        np.testing.assert_array_equal(upload["data"], update[:, 1::2])
        self.assertEqual(upload["offset"], (1, 2))
        self.assertFalse(upload["copy"])
        self.assertEqual(visual.node.update_count, 1)

    def test_update_between_display_samples_needs_no_gpu_upload(self):
        viewer, layer, visual = self.setup_visual((6, 10), (6, 5))
        enable_fast_texture_updates(viewer, layer)

        layer.events.labels_update.emit(
            data=np.array([[7]], dtype=np.uint8),
            offset=(2, 3),
        )

        self.assertEqual(visual.original_count, 0)
        self.assertFalse(visual.node._texture.uploads)

    def test_normal_sized_texture_keeps_partial_update_semantics(self):
        viewer, layer, visual = self.setup_visual((6, 10), (6, 10))
        enable_fast_texture_updates(viewer, layer)
        update = np.array([[2, 3], [4, 5]], dtype=np.uint8)

        layer.events.labels_update.emit(data=update, offset=(3, 4))

        upload = visual.node._texture.uploads[0]
        np.testing.assert_array_equal(upload["data"], update)
        self.assertEqual(upload["offset"], (3, 4))

    def test_unknown_texture_layout_uses_napari_fallback(self):
        viewer, layer, visual = self.setup_visual((6, 10), (4, 4))
        enable_fast_texture_updates(viewer, layer)

        layer.events.labels_update.emit(
            data=np.ones((2, 2), dtype=np.uint8),
            offset=(1, 1),
        )

        self.assertEqual(visual.original_count, 1)
        self.assertFalse(visual.node._texture.uploads)

    def test_texture_installation_is_idempotent(self):
        viewer, layer, visual = self.setup_visual((6, 10), (6, 5))

        self.assertTrue(enable_fast_texture_updates(viewer, layer))
        self.assertTrue(enable_fast_texture_updates(viewer, layer))

        self.assertIs(
            visual._on_partial_labels_update.__func__,
            fast_partial_labels_update,
        )
        self.assertEqual(len(layer.events.labels_update.callbacks), 1)


class FakeDrawable:
    def __init__(self):
        self.visible = False
        self.calls = []

    def set_data(self, **kwargs):
        self.calls.append(kwargs)


class FakePolygonVisual:
    def __init__(self, layer, overlay):
        self.layer = layer
        self.overlay = overlay
        self._dims_displayed = (0, 1)
        self._line = FakeDrawable()
        self._line.method = "agg"
        self._polygon = FakeDrawable()
        self._polygon.border = SimpleNamespace(method="agg")
        self._nodes = FakeDrawable()
        self._nodes_kwargs = {
            "face_color": (1, 1, 1, 1),
            "size": 8.0,
            "edge_width": 1.0,
            "edge_color": (0, 0, 0, 1),
        }
        self.original_count = 0

    def _on_points_change(self, event=None):
        del event
        self.original_count += 1


class FakeOverlay:
    def __init__(self):
        self.points = []
        self.events = SimpleNamespace(points=FakeEmitter())


class FastPolygonPreviewTest(unittest.TestCase):
    def setup_overlay(self):
        layer = FakeLayer((20, 30))
        overlay = FakeOverlay()
        layer._overlays["polygon"] = overlay
        labels_visual = FakeLabelsVisual(layer, (20, 30))
        polygon_visual = FakePolygonVisual(layer, overlay)
        overlay.events.points.connect(polygon_visual._on_points_change)
        viewer = make_viewer(layer, labels_visual, polygon_visual)
        return viewer, layer, overlay, polygon_visual

    def test_polygon_preview_is_visible_fast_outline_with_large_nodes(self):
        viewer, layer, overlay, visual = self.setup_overlay()
        self.assertTrue(optimize_polygon_preview(viewer, layer))

        overlay.points = [(1, 2), (3, 4), (5, 6)]
        overlay.events.points.emit()

        self.assertEqual(visual.original_count, 0)
        self.assertFalse(visual._polygon.visible)
        self.assertTrue(visual._line.visible)
        self.assertEqual(visual._line.method, "gl")
        self.assertEqual(visual._polygon.border.method, "gl")
        self.assertEqual(visual._nodes_kwargs["size"], 14.0)
        self.assertEqual(visual._nodes_kwargs["edge_width"], 2.0)

        expected_points = np.array(
            [[2, 1], [4, 3], [6, 5], [2, 1]]
        )
        np.testing.assert_array_equal(
            visual._line.calls[-1]["pos"], expected_points
        )
        self.assertEqual(visual._line.calls[-1]["width"], 3.0)

    def test_preview_compensates_for_oversized_texture_scale(self):
        viewer, layer, overlay, visual = self.setup_overlay()
        layer._transforms["tile2data"].scale = np.array([1.0, 2.0])
        optimize_polygon_preview(viewer, layer)

        overlay.points = [(10, 20), (30, 40), (50, 60)]
        overlay.events.points.emit()

        # VisPy receives x/y coordinates. Its parent Labels visual applies a
        # 2x x scale plus napari's pixel-center offsets, so the preview uses
        # the exact inverse transform.
        expected_points = np.array(
            [
                [9.75, 10],
                [19.75, 30],
                [29.75, 50],
                [9.75, 10],
            ],
            dtype=float,
        )
        np.testing.assert_array_equal(
            visual._line.calls[-1]["pos"], expected_points
        )

    def test_polygon_installation_is_idempotent(self):
        viewer, layer, overlay, visual = self.setup_overlay()

        self.assertTrue(optimize_polygon_preview(viewer, layer))
        self.assertTrue(optimize_polygon_preview(viewer, layer))

        self.assertIs(
            visual._on_points_change.__func__,
            fast_polygon_points_change,
        )
        self.assertEqual(len(overlay.events.points.callbacks), 1)


if __name__ == "__main__":
    unittest.main()
