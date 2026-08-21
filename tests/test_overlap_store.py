import io
import unittest
from unittest.mock import patch

import numpy as np

from napari_histo_label_editor._overlap_store import (
    FORMAT_VERSION,
    OverlapStore,
    hash_projection,
)


def rewrite_payload(payload: bytes, **replacements) -> bytes:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {
            key: np.array(archive[key], copy=True) for key in archive.files
        }
    arrays.update(replacements)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


class OverlapStoreTest(unittest.TestCase):
    def make_store(self):
        labels = np.array(
            [
                [0, 1, 1, 0, 2, 2, 0, 0, 1, 0],
                [0, 0, 2, 2, 2, 0, 0, 1, 1, 0],
                [1, 1, 0, 0, 0, 0, 2, 2, 0, 0],
            ],
            dtype=np.uint8,
        )
        store = OverlapStore.from_legacy(
            labels,
            {0: "Background", 1: "Tumor", 2: "Stroma", 3: "Review"},
            {1: "red", 2: "#00ff00", 3: "navy"},
            z_order=(1, 2, 3),
        )
        return labels, store

    @staticmethod
    def store_snapshot(store):
        return {
            "class_map": store.class_map,
            "class_colors": store.class_colors,
            "class_values": store.class_values,
            "z_order": store.z_order,
            "projection_dtype": store.projection_dtype,
            "projection_sha256": store.projection_sha256,
            "generation": store.generation,
            "packed_masks": np.array(store.packed_masks, copy=True),
            "projection": store.project(),
        }

    def assert_store_matches_snapshot(self, store, snapshot):
        self.assertEqual(store.class_map, snapshot["class_map"])
        self.assertEqual(store.class_colors, snapshot["class_colors"])
        self.assertEqual(store.class_values, snapshot["class_values"])
        self.assertEqual(store.z_order, snapshot["z_order"])
        self.assertEqual(store.projection_dtype, snapshot["projection_dtype"])
        self.assertEqual(
            store.projection_sha256,
            snapshot["projection_sha256"],
        )
        self.assertEqual(store.generation, snapshot["generation"])
        np.testing.assert_array_equal(
            store.packed_masks,
            snapshot["packed_masks"],
        )
        np.testing.assert_array_equal(store.project(), snapshot["projection"])

    def test_legacy_migration_preserves_projection_and_complete_metadata(self):
        labels, store = self.make_store()

        np.testing.assert_array_equal(store.project(), labels)
        self.assertEqual(store.shape, labels.shape)
        self.assertEqual(store.class_values, (1, 2, 3))
        self.assertEqual(
            store.class_map,
            {0: "Background", 1: "Tumor", 2: "Stroma", 3: "Review"},
        )
        self.assertEqual(
            store.class_colors,
            {1: "#ff0000", 2: "#00ff00", 3: "#000080"},
        )
        self.assertEqual(store.z_order, (1, 2, 3))
        self.assertEqual(store.packed_masks.shape, (3, 3, 2))
        self.assertFalse(store.packed_masks.flags.writeable)
        self.assertEqual(store.projection_sha256, hash_projection(labels))

    def test_independent_class_patch_preserves_hidden_membership(self):
        labels, store = self.make_store()

        changed = store.update_patch(2, (0, 1), np.ones((1, 2), dtype=bool))

        self.assertEqual(changed, 2)
        self.assertEqual(store.memberships_at(0, 1), (1, 2))
        self.assertEqual(store.memberships((0, 2)), (1, 2))
        self.assertEqual(store.project()[0, 1], 2)
        self.assertEqual(store.project(exclude=2)[0, 1], 1)

        changed = store.update_patch(2, (0, 1), np.zeros((1, 2), dtype=bool))

        self.assertEqual(changed, 2)
        self.assertEqual(store.memberships_at(0, 1), (1,))
        self.assertEqual(store.project()[0, 1], 1)
        np.testing.assert_array_equal(store.select_plane(1), labels == 1)

    def test_sparse_raise_and_restore_change_only_authoritative_tops(self):
        _, store = self.make_store()
        store.update_patch(2, (0, 1), np.ones((1, 2), dtype=bool))
        self.assertEqual(store.project()[0, 1], 2)
        self.assertEqual(store.project()[0, 2], 2)

        changed = store.raise_projection_indices(
            1,
            (np.array([0]), np.array([1])),
        )

        self.assertEqual(changed, 1)
        self.assertEqual(store.project()[0, 1], 1)
        self.assertEqual(store.project()[0, 2], 2)
        self.assertEqual(store.memberships_at(0, 1), (2, 1))
        self.assertEqual(
            store.restore_projection_indices(
                (np.array([0]), np.array([1])),
                np.array([2], dtype=np.uint8),
            ),
            1,
        )
        self.assertEqual(store.project()[0, 1], 2)

    def test_sparse_semantic_erase_reveals_and_restores_mixed_overlaps(self):
        _, store = self.make_store()
        store.update_patch(2, (0, 1), np.ones((1, 2), dtype=np.uint8))
        store.update_patch(3, (0, 1), np.ones((1, 1), dtype=np.uint8))
        indices = (np.array([0, 0, 0]), np.array([0, 1, 2]))

        with patch.object(
            store,
            "select_plane",
            side_effect=AssertionError("semantic erase unpacked a plane"),
        ):
            changed, top_before, top_after = store.erase_visible_indices(
                indices
            )

        self.assertEqual(changed, 2)
        np.testing.assert_array_equal(top_before, np.array([0, 3, 2]))
        np.testing.assert_array_equal(top_after, np.array([0, 2, 1]))
        self.assertEqual(store.memberships_at(0, 0), ())
        self.assertEqual(store.memberships_at(0, 1), (1, 2))
        self.assertEqual(store.memberships_at(0, 2), (1,))

        self.assertEqual(
            store.restore_erased_indices(
                indices,
                top_before,
                top_before,
                present=True,
            ),
            2,
        )
        self.assertEqual(store.memberships_at(0, 1), (1, 2, 3))
        self.assertEqual(store.memberships_at(0, 2), (1, 2))
        np.testing.assert_array_equal(store.project()[0, :3], top_before)

        self.assertEqual(
            store.restore_erased_indices(
                indices,
                top_before,
                top_after,
                present=False,
            ),
            2,
        )
        np.testing.assert_array_equal(store.project()[0, :3], top_after)

    def test_sparse_semantic_erase_rolls_back_every_failed_setter(self):
        indices = (np.array([0, 0, 0]), np.array([0, 1, 2]))
        for fail_at in (1, 2):
            with self.subTest(fail_at=fail_at):
                store = OverlapStore.from_legacy(
                    np.array([[1, 2, 3]], dtype=np.uint8),
                    {0: "Background", 1: "A", 2: "B", 3: "C"},
                )
                snapshot = self.store_snapshot(store)
                native_setter = store._set_membership_at_indices
                calls = 0

                def failing_setter(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == fail_at:
                        raise MemoryError("forced sparse setter failure")
                    return native_setter(*args, **kwargs)

                with patch.object(
                    store,
                    "_set_membership_at_indices",
                    side_effect=failing_setter,
                ):
                    with self.assertRaisesRegex(MemoryError, "forced sparse"):
                        store.erase_visible_indices(indices)

                self.assert_store_matches_snapshot(store, snapshot)

    def test_sparse_semantic_history_rolls_back_every_failed_setter(self):
        indices = (np.array([0, 0, 0]), np.array([0, 1, 2]))
        top_before = np.array([1, 2, 3], dtype=np.uint8)
        top_after = np.zeros(3, dtype=np.uint8)
        for fail_at in (1, 2):
            with self.subTest(fail_at=fail_at):
                store = OverlapStore.from_legacy(
                    top_before[np.newaxis, :],
                    {0: "Background", 1: "A", 2: "B", 3: "C"},
                )
                store.erase_visible_indices(indices)
                snapshot = self.store_snapshot(store)
                native_setter = store._set_membership_at_indices
                calls = 0

                def failing_setter(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == fail_at:
                        raise MemoryError("forced history setter failure")
                    return native_setter(*args, **kwargs)

                with patch.object(
                    store,
                    "_set_membership_at_indices",
                    side_effect=failing_setter,
                ):
                    with self.assertRaisesRegex(MemoryError, "forced history"):
                        store.restore_erased_indices(
                            indices,
                            top_before,
                            top_before,
                            present=True,
                        )

                self.assert_store_matches_snapshot(store, snapshot)
                np.testing.assert_array_equal(store.project()[0], top_after)

    def test_patch_crossing_byte_boundary_does_not_touch_neighbors(self):
        _, store = self.make_store()
        before = store.select_plane(3)

        selected = store.select_patch(1, (0, 7), (2, 3))
        np.testing.assert_array_equal(selected, store.select_plane(1)[0:2, 7:10])
        selected[...] = False
        np.testing.assert_array_equal(
            store.select_patch(1, (0, 7), (2, 3)),
            store.select_plane(1)[0:2, 7:10],
        )

        changed = store.update_patch(
            3,
            (1, 6),
            np.array([[1, 1, 1]], dtype=np.uint8),
        )

        expected = before.copy()
        expected[1, 6:9] = True
        self.assertEqual(changed, 3)
        np.testing.assert_array_equal(store.select_plane(3), expected)
        np.testing.assert_array_equal(
            store.select_plane(1),
            store.project(exclude=(2, 3)) == 1,
        )

    def test_update_plane_counts_only_changed_pixels(self):
        _, store = self.make_store()
        plane = store.select_plane(1)
        plane[0, 0] = True
        plane[0, 1] = False

        self.assertEqual(store.update_plane(1, plane), 2)
        self.assertEqual(store.update_plane(1, plane), 0)
        self.assertEqual(store.memberships_at(0, 0), (1,))
        self.assertNotIn(1, store.memberships_at(0, 1))

        plane[0, 0] = False
        plane[0, 1] = True
        self.assertEqual(store.update_plane_counts(1, plane), (2, 1, 1))

    def test_project_patch_matches_same_region_of_complete_projection(self):
        _, store = self.make_store()
        store.update_patch(3, (0, 7), np.ones((2, 2), dtype=bool))

        expected = store.project(exclude=2)[0:2, 6:10]
        actual = store.project_patch(
            (0, 6),
            (2, 4),
            exclude=2,
        )

        np.testing.assert_array_equal(actual, expected)

    def test_count_class_is_exact(self):
        labels, store = self.make_store()
        self.assertEqual(store.count_class(1), int(np.count_nonzero(labels == 1)))

    def test_remove_with_replacement_ors_membership_and_keeps_other_planes(self):
        labels, store = self.make_store()
        tumor = store.select_plane(1)
        stroma = store.select_plane(2)

        source_count = store.remove_class(2, replacement=1)

        self.assertEqual(source_count, int(np.count_nonzero(stroma)))
        self.assertEqual(store.class_values, (1, 3))
        self.assertNotIn(2, store.class_map)
        np.testing.assert_array_equal(store.select_plane(1), tumor | stroma)
        self.assertEqual(store.memberships_at(0, 4), (1,))
        self.assertEqual(store.project()[0, 4], 1)
        self.assertEqual(store.project()[0, 0], labels[0, 0])

    def test_payload_round_trip_restores_overlaps_order_names_and_colors(self):
        _, store = self.make_store()
        store.update_patch(2, (0, 1), np.ones((1, 2), dtype=bool))
        store.update_patch(3, (0, 1), np.array([[1]], dtype=bool))
        store.set_class_metadata(2, name="Fibrosis", color="orange")
        projection = store.project(dtype=np.uint16)

        payload = store.to_payload(projection)
        restored = OverlapStore.from_payload(payload, projection)

        self.assertGreater(len(payload), 0)
        self.assertEqual(restored.class_map[2], "Fibrosis")
        self.assertEqual(restored.class_colors[2], "#ffa500")
        self.assertEqual(restored.z_order, (1, 2, 3))
        self.assertEqual(restored.memberships_at(0, 1), (1, 2, 3))
        np.testing.assert_array_equal(restored.packed_masks, store.packed_masks)
        np.testing.assert_array_equal(restored.project(), projection)
        # The outer hash remains bound to the uint16 saved image while the
        # runtime top projection compacts to the smallest safe dtype.
        self.assertEqual(restored.projection_dtype, np.dtype(np.uint8))
        self.assertEqual(restored.projection_sha256, hash_projection(projection))

    def test_payload_can_be_validated_without_external_projection(self):
        _, store = self.make_store()
        projection = store.project()

        restored = OverlapStore.from_payload(store.to_payload(projection))

        np.testing.assert_array_equal(restored.project(), projection)

    def test_payload_rejects_projection_mismatch_without_losing_hidden_state(self):
        _, store = self.make_store()
        store.update_patch(2, (0, 1), np.ones((1, 1), dtype=bool))
        projection = store.project()
        payload = store.to_payload(projection)
        mismatched = projection.copy()
        mismatched[0, 1] = 1

        with self.assertRaisesRegex(ValueError, "hash does not match"):
            OverlapStore.from_payload(payload, mismatched)

    def test_payload_rejects_background_where_membership_exists(self):
        _, store = self.make_store()
        projection = store.project()
        payload = store.to_payload(projection)
        invalid_top = projection.copy()
        invalid_top[0, 1] = 0

        with self.assertRaisesRegex(ValueError, "top values"):
            OverlapStore.from_payload(payload, invalid_top)

    def test_payload_rejects_checksum_valid_projection_value_that_would_wrap(self):
        labels = np.array([[1]], dtype=np.uint16)
        store = OverlapStore.from_legacy(
            labels,
            {0: "Background", 1: "Tumor"},
        )
        wrapped_source = np.array([[257]], dtype=np.uint16)
        payload = rewrite_payload(
            store.to_payload(labels),
            projection_sha256=np.asarray(hash_projection(wrapped_source)),
        )

        with self.assertRaisesRegex(ValueError, "top values"):
            OverlapStore.from_payload(payload, wrapped_source)

    def test_payload_strictly_rejects_version_keys_types_and_padding(self):
        _, store = self.make_store()
        projection = store.project()
        payload = store.to_payload(projection)

        bad_version = rewrite_payload(
            payload,
            format_version=np.asarray(FORMAT_VERSION + 1, dtype=np.uint16),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            OverlapStore.from_payload(bad_version, projection)

        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            missing_arrays = {
                key: np.array(archive[key], copy=True)
                for key in archive.files
                if key != "class_names"
            }
        stream = io.BytesIO()
        np.savez_compressed(stream, **missing_arrays)
        with self.assertRaisesRegex(ValueError, "payload keys"):
            OverlapStore.from_payload(stream.getvalue(), projection)

        bad_shape_type = rewrite_payload(
            payload,
            shape=np.asarray(store.shape, dtype=np.int32),
        )
        with self.assertRaisesRegex(ValueError, "shape must be"):
            OverlapStore.from_payload(bad_shape_type, projection)

        packed = np.array(store.packed_masks, copy=True)
        packed[0, 0, -1] |= np.uint8(0b10000000)
        bad_padding = rewrite_payload(payload, packed_masks=packed)
        with self.assertRaisesRegex(ValueError, "padding bits"):
            OverlapStore.from_payload(bad_padding, projection)

    def test_add_empty_class_promotes_projection_dtype_and_round_trips(self):
        labels, store = self.make_store()

        store.add_class(300, "High value", "#abcdef")

        self.assertEqual(store.projection_dtype, np.dtype("<u2"))
        self.assertEqual(store.select_plane(300).sum(), 0)
        np.testing.assert_array_equal(store.project(), labels)
        payload = store.to_payload(store.project())
        restored = OverlapStore.from_payload(payload, store.project())
        self.assertEqual(restored.class_map[300], "High value")
        self.assertEqual(restored.class_colors[300], "#abcdef")

    def test_add_class_projection_oom_is_an_exact_no_op(self):
        _, store = self.make_store()
        snapshot = self.store_snapshot(store)

        with patch(
            "napari_histo_label_editor._overlap_store._copy_projection",
            side_effect=MemoryError("simulated projection promotion OOM"),
        ), self.assertRaises(MemoryError):
            store.add_class(300, "High value", "#abcdef")

        self.assert_store_matches_snapshot(store, snapshot)

    def test_remove_class_packed_allocation_oom_is_an_exact_no_op(self):
        _, store = self.make_store()
        snapshot = self.store_snapshot(store)

        with patch(
            "napari_histo_label_editor._overlap_store.np.delete",
            side_effect=MemoryError("simulated packed removal OOM"),
        ), self.assertRaises(MemoryError):
            store.remove_class(1, replacement=2)

        self.assert_store_matches_snapshot(store, snapshot)

    def test_remove_class_projection_oom_is_an_exact_no_op(self):
        _, store = self.make_store()
        snapshot = self.store_snapshot(store)

        with patch(
            "napari_histo_label_editor._overlap_store._copy_projection",
            side_effect=MemoryError("simulated projection copy OOM"),
        ), self.assertRaises(MemoryError):
            store.remove_class(1, replacement=2)

        self.assert_store_matches_snapshot(store, snapshot)

    def test_rejects_invalid_planes_patches_and_class_operations(self):
        _, store = self.make_store()

        with self.assertRaisesRegex(ValueError, "shape"):
            store.update_plane(1, np.zeros((2, 2), dtype=bool))
        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            store.update_patch(1, (0, 0), np.array([[2]], dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            store.update_patch(1, (2, 9), np.ones((2, 2), dtype=bool))
        with self.assertRaisesRegex(ValueError, "itself"):
            store.remove_class(1, replacement=1)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            store.select_plane(999)
        with self.assertRaises(IndexError):
            store.memberships_at(-1, 0)


if __name__ == "__main__":
    unittest.main()
