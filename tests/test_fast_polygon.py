import unittest
from unittest.mock import patch

import numpy as np
from napari.layers import Labels
from skimage.draw import polygon2mask as skimage_polygon2mask

from napari_histo_label_editor._fast_polygon import (
    enable_fast_polygon,
    fast_paint_polygon,
)


class FastPolygonTest(unittest.TestCase):
    def make_layer(self, shape=(32, 40)):
        layer = Labels(np.zeros(shape, dtype=np.uint8))
        layer.n_edit_dimensions = 2
        return layer

    def assert_matches_native(self, points, new_label=4):
        native = self.make_layer()
        fast = self.make_layer()

        native.paint_polygon(points, new_label)
        fast_paint_polygon(fast, points, new_label)

        np.testing.assert_array_equal(fast.data, native.data)

    def test_matches_native_for_internal_polygon(self):
        self.assert_matches_native([(4, 5), (4, 17), (14, 10)])

    def test_matches_native_for_polygon_clipped_at_image_boundaries(self):
        self.assert_matches_native([(-6, 3), (8, 44), (22, 5)])

    def test_polygon_entirely_outside_is_a_noop(self):
        layer = self.make_layer()
        fast_paint_polygon(layer, [(-8, -8), (-8, -3), (-3, -5)], 2)

        self.assertFalse(np.any(layer.data))
        self.assertFalse(layer._undo_history)

    def test_undo_restores_polygon(self):
        layer = self.make_layer()
        fast_paint_polygon(layer, [(3, 3), (3, 9), (9, 6)], 7)
        self.assertTrue(np.any(layer.data == 7))

        layer.undo()

        self.assertFalse(np.any(layer.data))

    def test_mask_allocation_is_limited_to_polygon_bounding_box(self):
        layer = self.make_layer((2048, 2048))
        allocated_shapes = []

        def recording_polygon2mask(shape, points):
            allocated_shapes.append(tuple(shape))
            return skimage_polygon2mask(shape, points)

        with patch(
            "napari_histo_label_editor._fast_polygon.polygon2mask",
            side_effect=recording_polygon2mask,
        ):
            fast_paint_polygon(
                layer,
                [(1000, 1000), (1000, 1010), (1008, 1005)],
                3,
            )

        self.assertEqual(allocated_shapes, [(9, 11)])

    def test_enable_fast_polygon_is_idempotent(self):
        layer = self.make_layer()
        enable_fast_polygon(layer)
        first = layer.paint_polygon
        enable_fast_polygon(layer)

        self.assertIs(layer.paint_polygon.__func__, fast_paint_polygon)
        self.assertIs(layer.paint_polygon.__func__, first.__func__)


if __name__ == "__main__":
    unittest.main()
