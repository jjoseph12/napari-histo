import unittest

import numpy as np

from napari_histo_label_editor._object_selection import (
    MAX_EXACT_OUTLINE_PIXELS,
    MAX_SELECTION_PIXELS,
    MAX_SELECTION_SEARCH_PIXELS,
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

    def test_selection_larger_than_old_limit_is_exact_and_compact(self):
        projection = np.ones((800, 800), dtype=np.uint8)

        selected = select_visible_component(
            projection,
            (400, 400),
            max_exact_outline_pixels=64,
        )

        self.assertEqual(selected.pixel_count, 640_000)
        self.assertEqual(selected.rows.dtype, np.dtype(np.int32))
        self.assertEqual(selected.columns.dtype, np.dtype(np.int32))
        self.assertFalse(selected.rows.flags.writeable)
        self.assertFalse(selected.columns.flags.writeable)
        self.assertLessEqual(
            selected.rows.nbytes + selected.columns.nbytes,
            selected.pixel_count * 8,
        )
        self.assertEqual(selected.bounds, (0, 800, 0, 800))
        self.assertTrue(selected.simplified_preview)

    def test_search_budget_is_independent_from_outline_budget(self):
        projection = np.zeros((2, 4096), dtype=np.uint8)
        projection[0] = 9

        selected = select_visible_component(
            projection,
            (0, 2048),
            max_selection_pixels=5000,
            max_search_pixels=8192,
            max_exact_outline_pixels=32,
        )

        self.assertEqual(selected.pixel_count, 4096)
        self.assertEqual(selected.bounds, (0, 1, 0, 4096))
        self.assertTrue(selected.simplified_preview)

    def test_exact_indices_can_be_consumed_in_bounded_views(self):
        projection = np.ones((9, 11), dtype=np.uint8)
        selected = select_visible_component(projection, (4, 5))

        chunks = tuple(selected.iter_indices(max_pixels=17))

        self.assertEqual(len(chunks), 6)
        self.assertTrue(all(rows.size <= 17 for rows, _ in chunks))
        np.testing.assert_array_equal(
            np.concatenate([rows for rows, _ in chunks]),
            selected.rows,
        )
        np.testing.assert_array_equal(
            np.concatenate([columns for _, columns in chunks]),
            selected.columns,
        )
        with self.assertRaisesRegex(ValueError, "max_pixels"):
            tuple(selected.iter_indices(0))

    def test_very_large_path_uses_immutable_runs_and_keeps_delete_indices(self):
        projection = np.ones((6, 10), dtype=np.uint8)

        selected = select_visible_component(
            projection,
            (3, 5),
            max_selection_pixels=projection.size,
            max_search_pixels=projection.size,
            max_sparse_pixels=8,
            max_run_bytes=1024,
        )

        self.assertIsNotNone(selected.runs)
        self.assertIs(selected.indices, selected.runs)
        self.assertEqual(selected.pixel_count, projection.size)
        self.assertEqual(selected.bounds, (0, 6, 0, 10))
        self.assertEqual(selected.rows.size, 0)
        self.assertEqual(selected.columns.size, 0)
        self.assertFalse(selected.runs.rows.flags.writeable)
        self.assertFalse(selected.runs.starts.flags.writeable)
        self.assertFalse(selected.runs.stops.flags.writeable)
        chunks = tuple(selected.iter_indices(max_pixels=13))
        self.assertTrue(all(rows.size <= 13 for rows, _ in chunks))
        coordinates = set()
        for rows, columns in chunks:
            coordinates.update(zip(rows.tolist(), columns.tolist()))
        self.assertEqual(
            coordinates,
            set(zip(*np.nonzero(projection))),
        )
        self.assertTrue(selected.simplified_preview)
        with self.assertRaisesRegex(ValueError, "very large run-backed"):
            selected.translated(1, 0, projection.shape)

    def test_search_window_limit_has_distinct_bounded_error(self):
        projection = np.zeros((2, 4096), dtype=np.uint8)
        projection[0] = 3

        with self.assertRaisesRegex(
            SelectionTooLargeError,
            "2,048-pixel search window",
        ):
            select_visible_component(
                projection,
                (0, 2048),
                max_selection_pixels=5000,
                max_search_pixels=2048,
                max_exact_outline_pixels=32,
            )

    def test_default_budgets_cover_common_full_slide_canvas(self):
        self.assertEqual(MAX_EXACT_OUTLINE_PIXELS, 4 * 1024 * 1024)
        self.assertGreaterEqual(MAX_SELECTION_PIXELS, 16_000_000)
        self.assertGreaterEqual(
            MAX_SELECTION_SEARCH_PIXELS,
            7048 * 8001,
        )

    def test_default_large_span_keeps_exact_pixels_with_bounded_preview(self):
        projection = np.zeros((2049, 2048), dtype=np.uint8)
        projection[[0, -1], :] = 6
        projection[:, [0, -1]] = 6

        selected = select_visible_component(projection, (0, 0))

        self.assertEqual(selected.pixel_count, 2 * 2049 + 2 * 2048 - 4)
        self.assertEqual(selected.bounds, (0, 2049, 0, 2048))
        self.assertTrue(selected.simplified_preview)
        np.testing.assert_array_equal(
            selected.outline,
            np.array(
                [
                    [-0.5, -0.5],
                    [-0.5, 2047.5],
                    [2048.5, 2047.5],
                    [2048.5, -0.5],
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
        self.assertFalse(moved.rows.flags.writeable)
        self.assertFalse(moved.columns.flags.writeable)
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
