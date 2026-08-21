import unittest
from unittest.mock import patch

import numpy as np
from napari.layers import Labels
from scipy import ndimage as ndi
from skimage.segmentation import flood as reference_flood

from napari_histo_label_editor._fast_fill import (
    bounded_flood_indices,
    bounded_overlap_flood_indices,
    enable_fast_fill,
    enable_overlap_fill,
    fast_fill,
    overlap_fill,
)


class FastFillTest(unittest.TestCase):
    def make_layer(self, data):
        layer = Labels(np.asarray(data))
        layer.n_edit_dimensions = 2
        return layer

    def test_connected_fill_matches_native_four_connected_semantics(self):
        data = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.uint8,
        )
        native = self.make_layer(data.copy())
        fast = self.make_layer(data.copy())
        native.contiguous = True
        fast.contiguous = True

        native.fill((1, 1), 7)
        fast_fill(fast, (1, 1), 7)

        np.testing.assert_array_equal(fast.data, native.data)
        self.assertEqual(fast.data[0, 0], 1)  # diagonal is disconnected
        self.assertEqual(fast.data[2, 3], 1)  # separate component

    def test_noncontiguous_fill_matches_native_global_replacement(self):
        data = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        native = self.make_layer(data.copy())
        fast = self.make_layer(data.copy())
        native.contiguous = False
        fast.contiguous = False

        native.fill((0, 0), 4)
        fast_fill(fast, (0, 0), 4)

        np.testing.assert_array_equal(fast.data, native.data)
        np.testing.assert_array_equal(
            fast.data,
            np.array([[4, 0, 4], [0, 4, 0]], dtype=np.uint8),
        )

    def test_undo_restores_a_connected_fill(self):
        layer = self.make_layer(
            np.array([[0, 2, 2], [0, 2, 0]], dtype=np.uint8)
        )

        fast_fill(layer, (0, 1), 5)
        self.assertEqual(np.count_nonzero(layer.data == 5), 3)
        layer.undo()

        np.testing.assert_array_equal(
            layer.data,
            np.array([[0, 2, 2], [0, 2, 0]], dtype=np.uint8),
        )

    def test_out_of_bounds_and_same_label_are_noops_without_history(self):
        layer = self.make_layer(np.array([[0, 2], [2, 2]], dtype=np.uint8))
        original = layer.data.copy()
        initial_history = len(layer._undo_history)

        fast_fill(layer, (-1, 0), 3)
        fast_fill(layer, (2, 0), 3)
        fast_fill(layer, (0, 1), 2)

        np.testing.assert_array_equal(layer.data, original)
        self.assertEqual(len(layer._undo_history), initial_history)

    def test_preserve_labels_matches_native_semantics(self):
        data = np.array([[0, 2, 2], [0, 3, 0]], dtype=np.uint8)
        native = self.make_layer(data.copy())
        fast = self.make_layer(data.copy())
        for layer in (native, fast):
            layer.selected_label = 5
            layer.preserve_labels = True

        # A non-background existing label must not be overwritten.
        native.fill((0, 1), 5)
        fast_fill(fast, (0, 1), 5)
        np.testing.assert_array_equal(fast.data, native.data)
        np.testing.assert_array_equal(fast.data, data)

        # Background remains fillable while preserve_labels is active.
        native.fill((0, 0), 5)
        fast_fill(fast, (0, 0), 5)
        np.testing.assert_array_equal(fast.data, native.data)

        # Switching to background remembers the prior selected label and
        # permits erasing that label, but no other foreground label.
        for layer in (native, fast):
            layer.selected_label = 5
            layer.selected_label = 0
        native.fill((0, 0), 0)
        fast_fill(fast, (0, 0), 0)
        native.fill((1, 1), 0)
        fast_fill(fast, (1, 1), 0)
        np.testing.assert_array_equal(fast.data, native.data)
        self.assertEqual(fast.data[1, 1], 3)

    def test_refresh_flag_is_forwarded_to_data_setitem(self):
        layer = self.make_layer(np.array([[1, 1], [0, 0]], dtype=np.uint8))

        with patch.object(layer, "data_setitem", wraps=layer.data_setitem) as setitem:
            fast_fill(layer, (0, 0), 3, refresh=False)

        self.assertFalse(setitem.call_args.args[2])

    def test_enable_fast_fill_is_idempotent(self):
        layer = self.make_layer(np.array([[1, 1], [0, 0]], dtype=np.uint8))

        enable_fast_fill(layer)
        first = layer.fill
        enable_fast_fill(layer)

        self.assertIs(layer.fill.__func__, fast_fill)
        self.assertIs(layer.fill.__func__, first.__func__)
        layer.fill((0, 0), 3)
        np.testing.assert_array_equal(
            layer.data,
            np.array([[3, 3], [0, 0]], dtype=np.uint8),
        )

    def test_connected_fill_does_not_use_scipy_component_labels(self):
        layer = self.make_layer(np.ones((32, 32), dtype=np.uint8))
        layer.contiguous = True

        # Native napari allocates a full-size int32 output in ndi.label.
        # The fast path must remain seed-based and avoid that allocation.
        with patch.object(
            ndi,
            "label",
            side_effect=AssertionError("full component labeling used"),
        ):
            fast_fill(layer, (0, 0), 2)

        self.assertTrue(np.all(layer.data == 2))

    def test_bounded_flood_matches_reference_across_expanding_windows(self):
        labels = np.zeros((256, 320), dtype=np.uint8)
        labels[30:220, 70:92] = 4
        labels[190:215, 70:280] = 4
        seed = (40, 80)

        expected = np.nonzero(
            reference_flood(labels, seed, connectivity=1)
        )
        actual = bounded_flood_indices(
            labels,
            seed,
            initial_window=32,
            max_local_pixels=labels.size,
        )

        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_growth_overshoot_is_clipped_instead_of_rejecting_component(self):
        labels = np.zeros((100, 100), dtype=np.uint8)
        labels[30:70, 30:70] = 5
        allocated_shapes = []

        def recording_flood(data, seed, connectivity):
            allocated_shapes.append(data.shape)
            return reference_flood(data, seed, connectivity=connectivity)

        with patch(
            "napari_histo_label_editor._fast_fill.flood",
            side_effect=recording_flood,
        ):
            actual = bounded_flood_indices(
                labels,
                (50, 50),
                initial_window=10,
                max_local_pixels=5000,
                allow_full_fallback=False,
            )

        expected = np.nonzero(
            reference_flood(labels, (50, 50), connectivity=1)
        )
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertTrue(
            all(height * width <= 5000 for height, width in allocated_shapes)
        )
        self.assertNotIn(labels.shape, allocated_shapes)

    def test_bounded_flood_matches_reference_for_random_components(self):
        rng = np.random.default_rng(2048)
        labels = rng.integers(0, 4, size=(80, 96), dtype=np.uint8)

        for seed in [(0, 0), (17, 23), (40, 48), (79, 95)]:
            expected = np.nonzero(
                reference_flood(labels, seed, connectivity=1)
            )
            actual = bounded_flood_indices(
                labels,
                seed,
                initial_window=15,
                max_local_pixels=labels.size,
            )
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])

    def test_component_can_leave_initial_window_and_reenter(self):
        labels = np.zeros((320, 360), dtype=np.uint8)
        labels[80:86, 80:290] = 6
        labels[80:270, 284:290] = 6
        labels[264:270, 35:290] = 6
        labels[155:270, 35:41] = 6
        labels[155:161, 35:120] = 6
        seed = (82, 100)

        expected = np.nonzero(
            reference_flood(labels, seed, connectivity=1)
        )
        actual = bounded_flood_indices(
            labels,
            seed,
            initial_window=32,
            max_local_pixels=labels.size,
        )

        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_small_component_never_allocates_a_full_slide_flood_map(self):
        labels = np.zeros((4096, 4096), dtype=np.uint8)
        labels[1900:2100, 1900:2100] = 3
        allocated_shapes = []

        def recording_flood(data, seed, connectivity):
            allocated_shapes.append(data.shape)
            return reference_flood(data, seed, connectivity=connectivity)

        with patch(
            "napari_histo_label_editor._fast_fill.flood",
            side_effect=recording_flood,
        ):
            indices = bounded_flood_indices(labels, (2000, 2000))

        self.assertEqual(len(indices[0]), 200 * 200)
        self.assertEqual(allocated_shapes, [(1024, 1024)])

    def test_large_component_uses_compiled_full_array_fallback(self):
        labels = np.ones((300, 400), dtype=np.uint8)
        allocated_shapes = []

        def recording_flood(data, seed, connectivity):
            allocated_shapes.append(data.shape)
            return reference_flood(data, seed, connectivity=connectivity)

        with patch(
            "napari_histo_label_editor._fast_fill.flood",
            side_effect=recording_flood,
        ):
            indices = bounded_flood_indices(
                labels,
                (150, 200),
                initial_window=32,
                max_local_pixels=2000,
            )

        self.assertEqual(len(indices[0]), labels.size)
        self.assertEqual(allocated_shapes[-1], labels.shape)

    def test_overlap_fill_follows_visible_semantic_component(self):
        composite = np.array(
            [
                [2, 2, 0, 3, 3],
                [2, 0, 0, 3, 0],
                [0, 0, 4, 4, 0],
            ],
            dtype=np.uint8,
        )
        layer = self.make_layer(np.zeros(composite.shape, dtype=np.uint8))
        layer.contiguous = True
        enable_overlap_fill(layer, composite, active_value=9)

        layer.fill((0, 0), 1)

        expected = np.zeros_like(composite)
        expected[0, :2] = 1
        expected[1, 0] = 1
        np.testing.assert_array_equal(layer.data, expected)
        self.assertIs(layer.fill.__func__, overlap_fill)

    def test_overlap_erase_follows_clicked_visible_semantic_component(self):
        composite = np.array(
            [
                [2, 2, 0, 0, 3, 3],
                [2, 0, 0, 0, 3, 0],
                [0, 0, 0, 0, 0, 0],
                [4, 4, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        active = np.array(
            [
                [1, 1, 0, 0, 1, 1],
                [1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        layer = self.make_layer(active.copy())
        layer.contiguous = True
        enable_overlap_fill(layer, composite, active_value=9)

        layer.fill((0, 0), 0)

        expected = active.copy()
        expected[0, :2] = 0
        expected[1, 0] = 0
        np.testing.assert_array_equal(layer.data, expected)

    def test_overlap_fill_crosses_hidden_active_membership(self):
        # The transparent active mask must not split the actually visible
        # class-2 component. Painting from the left reaches the right side.
        composite = np.full((1, 3), 2, dtype=np.uint8)
        active = np.array([[0, 1, 0]], dtype=np.uint8)
        layer = self.make_layer(active.copy())
        layer.contiguous = True
        enable_overlap_fill(layer, composite, active_value=9)

        layer.fill((0, 0), 1)

        np.testing.assert_array_equal(
            layer.data,
            np.ones((1, 3), dtype=np.uint8),
        )

    def test_overlap_fill_seeded_on_hidden_membership_paints_neighbors(self):
        # The clicked membership is already 1 in the transparent edit mask,
        # but class 2 is authoritative/visible there. It must not trigger the
        # ordinary binary-layer early return before the visible component is
        # flooded.
        composite = np.full((1, 3), 2, dtype=np.uint8)
        active = np.array([[0, 1, 0]], dtype=np.uint8)
        layer = self.make_layer(active.copy())
        layer.contiguous = True
        enable_overlap_fill(layer, composite, active_value=9)

        layer.fill((0, 1), 1)

        np.testing.assert_array_equal(
            layer.data,
            np.ones((1, 3), dtype=np.uint8),
        )

    def test_bounded_overlap_fill_materializes_only_local_object_window(self):
        composite = np.zeros((4096, 4096), dtype=np.uint8)
        composite[1900:2100, 1900:2100] = 3
        active = np.zeros_like(composite)
        allocated_shapes = []

        def recording_flood(data, seed, connectivity):
            allocated_shapes.append(data.shape)
            return reference_flood(data, seed, connectivity=connectivity)

        with patch(
            "napari_histo_label_editor._fast_fill.flood",
            side_effect=recording_flood,
        ):
            indices = bounded_overlap_flood_indices(
                composite,
                active,
                active_value=9,
                seed=(2000, 2000),
            )

        self.assertEqual(len(indices[0]), 200 * 200)
        self.assertEqual(allocated_shapes, [(1024, 1024)])


if __name__ == "__main__":
    unittest.main()
