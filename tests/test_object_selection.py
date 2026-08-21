import unittest

import numpy as np

from napari_histo_label_editor._object_selection import (
    SelectionTooLargeError,
    select_visible_component,
)


class ObjectSelectionTest(unittest.TestCase):
    def test_selects_only_clicked_four_connected_visible_component(self):
        projection = np.array(
            [
                [0, 1, 1, 0, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 2],
            ],
            dtype=np.uint8,
        )

        selected = select_visible_component(projection, (0, 1))

        self.assertEqual(selected.value, 1)
        self.assertEqual(selected.pixel_count, 3)
        self.assertEqual(selected.bounds, (0, 2, 1, 3))
        self.assertEqual(
            set(zip(selected.rows.tolist(), selected.columns.tolist())),
            {(0, 1), (0, 2), (1, 1)},
        )
        self.assertFalse(selected.simplified_preview)
        self.assertGreaterEqual(selected.outline.shape[0], 4)

    def test_background_and_outside_have_no_selection(self):
        projection = np.zeros((3, 4), dtype=np.uint8)
        self.assertIsNone(select_visible_component(projection, (1, 1)))
        self.assertIsNone(select_visible_component(projection, (-1, 1)))
        self.assertIsNone(select_visible_component(projection, (3, 1)))

    def test_diagonal_object_is_not_four_connected(self):
        projection = np.eye(3, dtype=np.uint8)
        selected = select_visible_component(projection, (0, 0))
        self.assertEqual(selected.pixel_count, 1)

    def test_large_outline_uses_bounded_rectangle_without_losing_pixels(self):
        projection = np.ones((4, 6), dtype=np.uint8)
        selected = select_visible_component(
            projection,
            (2, 3),
            max_exact_outline_pixels=8,
        )
        self.assertEqual(selected.pixel_count, 24)
        self.assertTrue(selected.simplified_preview)
        np.testing.assert_array_equal(
            selected.outline,
            np.array(
                [
                    [-0.5, -0.5],
                    [-0.5, 5.5],
                    [3.5, 5.5],
                    [3.5, -0.5],
                ]
            ),
        )

    def test_translation_is_exact_and_rejects_image_exit(self):
        projection = np.zeros((7, 8), dtype=np.uint8)
        projection[2:4, 3:5] = 7
        selected = select_visible_component(projection, (2, 3))

        moved = selected.translated(2, -1, projection.shape)

        self.assertEqual(moved.value, 7)
        np.testing.assert_array_equal(moved.rows, selected.rows + 2)
        np.testing.assert_array_equal(moved.columns, selected.columns - 1)
        np.testing.assert_allclose(
            moved.outline,
            selected.outline + np.array([2.0, -1.0]),
        )
        self.assertEqual(len(moved.outlines), len(selected.outlines))
        with self.assertRaisesRegex(ValueError, "outside the image"):
            selected.translated(-3, 0, projection.shape)

    def test_outline_vertex_count_is_bounded(self):
        projection = np.zeros((20, 20), dtype=np.uint8)
        projection[2:18, 2:18] = 1
        projection[4:16, 4:16] = 0
        projection[9, 4:16] = 1
        selected = select_visible_component(
            projection,
            (2, 2),
            max_outline_vertices=8,
        )
        self.assertLessEqual(
            sum(outline.shape[0] for outline in selected.outlines),
            8,
        )

    def test_hole_boundaries_are_available_for_display(self):
        projection = np.zeros((9, 9), dtype=np.uint8)
        projection[1:8, 1:8] = 3
        projection[3:6, 3:6] = 0

        selected = select_visible_component(projection, (1, 1))

        self.assertEqual(selected.value, 3)
        self.assertEqual(selected.pixel_count, 40)
        self.assertGreaterEqual(len(selected.outlines), 2)

    def test_selection_refuses_before_unbounded_component_coordinates(self):
        projection = np.ones((5, 5), dtype=np.uint8)

        with self.assertRaisesRegex(SelectionTooLargeError, "too large"):
            select_visible_component(
                projection,
                (2, 2),
                max_selection_pixels=8,
            )

    def test_simplified_outline_is_explicitly_marked_approximate(self):
        projection = np.ones((10, 10), dtype=np.uint8)

        selected = select_visible_component(
            projection,
            (5, 5),
            max_outline_vertices=4,
        )

        self.assertTrue(selected.simplified_preview)
        self.assertEqual(
            sum(outline.shape[0] for outline in selected.outlines),
            4,
        )


if __name__ == "__main__":
    unittest.main()
