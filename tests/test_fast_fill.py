import unittest
from unittest.mock import patch

import numpy as np
from napari.layers import Labels
from scipy import ndimage as ndi

from napari_histo_label_editor._fast_fill import enable_fast_fill, fast_fill


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


if __name__ == "__main__":
    unittest.main()
