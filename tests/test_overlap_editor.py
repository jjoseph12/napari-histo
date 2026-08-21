import unittest
from unittest.mock import patch

import numpy as np

from napari_histo_label_editor._overlap_editor import (
    OverlapEditorController,
)
from napari_histo_label_editor._overlap_store import OverlapStore


class _FailAfterProxyWrite(np.ndarray):
    """Fault-injection array that writes once and then raises."""

    fail_next_write = False

    def __new__(cls, values):
        return np.array(values, copy=True).view(cls)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.fail_next_write:
            self.fail_next_write = False
            raise MemoryError("forced active proxy failure")


class OverlapEditorControllerTest(unittest.TestCase):
    def make_store(self):
        labels = np.zeros((8, 9), dtype=np.uint8)
        labels[1:7, 1:7] = 1
        labels[3:6, 3:6] = 2
        store = OverlapStore.from_legacy(
            labels,
            {0: "Background", 1: "Lower", 2: "Upper", 3: "New"},
        )
        # Legacy masks are mutually exclusive. Add class 1 below class 2 so
        # the runtime begins with a genuine overlap.
        lower = store.select_plane(1)
        lower[3:6, 3:6] = True
        store.update_plane(1, lower)
        # Re-add the original class-2 plane so it is locally visible over the
        # newly hidden class-1 membership at the overlap.
        upper = store.select_plane(2)
        store.update_plane(2, np.zeros_like(upper))
        store.update_plane(2, upper)
        return store

    @staticmethod
    def make_move_store():
        labels = np.zeros((5, 9), dtype=np.uint8)
        labels[2, [1, 3, 4, 5, 6]] = 2
        store = OverlapStore.from_legacy(
            labels,
            {0: "Background", 1: "Moved", 2: "Lower", 3: "Other"},
        )
        store.update_patch(
            3,
            (2, 2),
            np.array([[1, 0, 1]], dtype=np.uint8),
        )
        store.update_patch(
            1,
            (2, 1),
            np.ones((1, 3), dtype=np.uint8),
        )
        return store

    def test_composite_includes_active_and_is_read_only(self):
        controller = OverlapEditorController(self.make_store(), 2)

        self.assertEqual(controller.active_class, 2)
        self.assertEqual(controller.composite[4, 4], 2)
        self.assertEqual(controller.edit_mask[4, 4], 1)
        self.assertEqual(controller.edit_mask.dtype, np.dtype(np.uint8))
        self.assertTrue(
            np.shares_memory(
                controller.composite,
                controller.store.projection_view,
            )
        )
        self.assertFalse(controller.composite.flags.writeable)
        with self.assertRaises(ValueError):
            controller.composite[4, 4] = 0

    def test_patch_add_and_erase_preserve_lower_memberships(self):
        store = self.make_store()
        controller = OverlapEditorController(store, 3)

        self.assertEqual(controller.composite[4, 4], 2)
        self.assertEqual(
            controller.process_patch(np.ones((1, 1), dtype=np.uint8), (4, 4)),
            1,
        )
        self.assertEqual(store.memberships_at(4, 4), (1, 2, 3))
        self.assertEqual(controller.composite[4, 4], 3)
        self.assertEqual(controller.saved_projection()[4, 4], 3)

        # Replaying the same patch is idempotent.
        self.assertEqual(
            controller.process_patch(np.ones((1, 1), dtype=np.uint8), (4, 4)),
            0,
        )
        controller.edit_mask[4, 4] = 0
        self.assertEqual(
            controller.process_indices((np.array([4]), np.array([4]))),
            1,
        )
        self.assertEqual(store.memberships_at(4, 4), (1, 2))
        self.assertEqual(controller.composite[4, 4], 2)

    def test_explicit_repaint_raises_hidden_active_only_at_touched_pixel(self):
        store = self.make_store()
        controller = OverlapEditorController(store, 1)
        unrelated = int(controller.composite[4, 5])

        changed, bounds = controller.raise_active_indices(
            (np.array([4]), np.array([4]))
        )

        self.assertEqual(changed, 1)
        self.assertEqual(bounds, (4, 5, 4, 5))
        self.assertEqual(controller.composite[4, 4], 1)
        self.assertEqual(controller.composite[4, 5], unrelated)
        self.assertEqual(store.memberships_at(4, 4), (2, 1))

    def test_three_class_erase_reveals_stable_hidden_fallbacks(self):
        store = self.make_store()
        controller = OverlapEditorController(store, 3)
        controller.process_patch(np.ones((1, 1), dtype=np.uint8), (4, 4))
        self.assertEqual(controller.composite[4, 4], 3)
        self.assertEqual(store.memberships_at(4, 4), (1, 2, 3))

        controller.process_patch(np.zeros((1, 1), dtype=np.uint8), (4, 4))
        self.assertEqual(controller.composite[4, 4], 2)
        controller.select_class(2)
        controller.process_patch(np.zeros((1, 1), dtype=np.uint8), (4, 4))
        self.assertEqual(controller.composite[4, 4], 1)
        self.assertEqual(store.memberships_at(4, 4), (1,))

    def test_semantic_erase_mixed_tops_preserves_hidden_active_and_restores(self):
        store = self.make_store()
        store.update_patch(3, (3, 3), np.ones((1, 1), dtype=np.uint8))
        controller = OverlapEditorController(store, 1)
        indices = (
            np.array([4, 2, 0, 3]),
            np.array([4, 2, 0, 3]),
        )

        with patch.object(
            store,
            "select_plane",
            side_effect=AssertionError("semantic erase unpacked a plane"),
        ), patch.object(
            store,
            "project",
            side_effect=AssertionError("semantic erase projected the slide"),
        ):
            changed, bounds, top_before, top_after = (
                controller.erase_visible_indices(indices)
            )

        self.assertEqual(changed, 3)
        self.assertEqual(bounds, (0, 5, 0, 5))
        np.testing.assert_array_equal(top_before, np.array([0, 1, 3, 2]))
        np.testing.assert_array_equal(top_after, np.array([0, 0, 2, 1]))
        self.assertEqual(store.memberships_at(4, 4), (1,))
        self.assertEqual(store.memberships_at(3, 3), (1, 2))
        self.assertEqual(controller.edit_mask[4, 4], 1)
        self.assertEqual(controller.edit_mask[3, 3], 1)
        self.assertEqual(controller.edit_mask[2, 2], 0)

        changed, _ = controller.restore_erased_indices(
            controller.normalize_indices(indices),
            top_before,
            top_before,
            present=True,
        )
        self.assertEqual(changed, 3)
        self.assertEqual(store.memberships_at(4, 4), (1, 2))
        self.assertEqual(store.memberships_at(3, 3), (1, 2, 3))
        self.assertEqual(controller.edit_mask[2, 2], 1)

        changed, _ = controller.restore_erased_indices(
            controller.normalize_indices(indices),
            top_before,
            top_after,
            present=False,
        )
        self.assertEqual(changed, 3)
        np.testing.assert_array_equal(
            controller.projection_values(indices),
            top_after,
        )

    def test_changed_patch_reads_raw_edit_mask(self):
        store = self.make_store()
        controller = OverlapEditorController(store, 3)
        controller.edit_mask[2:4, 5:7] = 1

        changed = controller.process_changed_patch((2, 5), (2, 2))

        self.assertEqual(changed, 4)
        np.testing.assert_array_equal(
            store.select_plane(3)[2:4, 5:7],
            np.ones((2, 2), dtype=bool),
        )

    def test_sparse_object_move_updates_active_proxy_and_replays_exactly(self):
        store = self.make_move_store()
        controller = OverlapEditorController(store, 1)
        source = (np.array([2, 2, 2]), np.array([1, 2, 3]))
        before_membership = controller.edit_mask.copy()
        before_projection = controller.composite.copy()
        revision = controller.revision

        with patch.object(
            store,
            "select_plane",
            side_effect=AssertionError("object move unpacked a plane"),
        ), patch.object(
            store,
            "project",
            side_effect=AssertionError("object move projected the slide"),
        ):
            delta = controller.plan_object_move(
                1,
                source,
                (0, 2),
                expected_revision=revision,
            )
            moved = controller.apply_overlap_delta(
                delta,
                forward=True,
                expected_revision=revision,
            )
            controller.apply_overlap_delta(delta, forward=False)

        self.assertEqual(moved.source_bounds, (2, 3, 1, 4))
        self.assertEqual(moved.destination_bounds, (2, 3, 3, 6))
        np.testing.assert_array_equal(controller.edit_mask, before_membership)
        np.testing.assert_array_equal(controller.composite, before_projection)

        controller.apply_overlap_delta(delta, forward=True)
        np.testing.assert_array_equal(
            np.flatnonzero(controller.edit_mask[2]),
            np.array([3, 4, 5]),
        )
        np.testing.assert_array_equal(
            controller.composite[2, 1:6],
            np.array([2, 3, 1, 1, 1]),
        )

    def test_object_move_does_not_switch_or_rewrite_other_active_proxy(self):
        store = self.make_move_store()
        controller = OverlapEditorController(store, 2)
        source = (np.array([2, 2, 2]), np.array([1, 2, 3]))
        active_before = controller.edit_mask.copy()

        delta = controller.plan_object_move(
            1,
            source,
            (0, 2),
            expected_revision=controller.revision,
        )
        controller.apply_overlap_delta(delta, forward=True)

        self.assertEqual(controller.active_class, 2)
        np.testing.assert_array_equal(controller.edit_mask, active_before)
        np.testing.assert_array_equal(
            np.flatnonzero(store.select_plane(1)[2]),
            np.array([3, 4, 5]),
        )

    def test_object_move_accepts_proxy_already_changed_by_native_history(self):
        store = self.make_move_store()
        controller = OverlapEditorController(store, 1)
        source = (np.array([2, 2, 2]), np.array([1, 2, 3]))
        delta = controller.plan_object_move(1, source, (0, 2))

        # Napari applies its binary history before the paired overlap tracker
        # restores packed membership and visible tops.
        controller.edit_mask[
            delta.affected_indices
        ] = delta.membership_after
        controller.apply_overlap_delta(delta, forward=True)
        np.testing.assert_array_equal(
            np.flatnonzero(store.select_plane(1)[2]),
            np.array([3, 4, 5]),
        )

        controller.edit_mask[
            delta.affected_indices
        ] = delta.membership_before
        controller.apply_overlap_delta(delta, forward=False)
        np.testing.assert_array_equal(
            np.flatnonzero(store.select_plane(1)[2]),
            np.array([1, 2, 3]),
        )

    def test_object_move_rejects_stale_selection_revision_without_changes(self):
        store = self.make_move_store()
        controller = OverlapEditorController(store, 1)
        source = (np.array([2, 2, 2]), np.array([1, 2, 3]))
        revision = controller.revision
        edit_before = controller.edit_mask.copy()
        projection_before = controller.composite.copy()
        store.set_class_metadata(3, name="Renamed")

        with self.assertRaisesRegex(RuntimeError, "stale"):
            controller.plan_object_move(
                1,
                source,
                (0, 2),
                expected_revision=revision,
            )

        np.testing.assert_array_equal(controller.edit_mask, edit_before)
        np.testing.assert_array_equal(controller.composite, projection_before)

    def test_object_move_proxy_failure_is_atomic_forward_and_backward(self):
        store = self.make_move_store()
        controller = OverlapEditorController(store, 1)
        source = (np.array([2, 2, 2]), np.array([1, 2, 3]))
        delta = controller.plan_object_move(1, source, (0, 2))

        original_packed = np.array(store.packed_masks, copy=True)
        original_projection = controller.composite.copy()
        original_proxy = controller.edit_mask.copy()
        original_revision = store.revision
        controller._edit_mask = _FailAfterProxyWrite(controller.edit_mask)
        controller._edit_mask.fail_next_write = True

        with self.assertRaisesRegex(MemoryError, "active proxy"):
            controller.apply_overlap_delta(delta, forward=True)

        np.testing.assert_array_equal(store.packed_masks, original_packed)
        np.testing.assert_array_equal(controller.composite, original_projection)
        np.testing.assert_array_equal(controller.edit_mask, original_proxy)
        self.assertEqual(store.revision, original_revision)

        controller.apply_overlap_delta(delta, forward=True)
        moved_packed = np.array(store.packed_masks, copy=True)
        moved_projection = controller.composite.copy()
        moved_proxy = controller.edit_mask.copy()
        moved_revision = store.revision
        controller._edit_mask.fail_next_write = True

        with self.assertRaisesRegex(MemoryError, "active proxy"):
            controller.apply_overlap_delta(delta, forward=False)

        np.testing.assert_array_equal(store.packed_masks, moved_packed)
        np.testing.assert_array_equal(controller.composite, moved_projection)
        np.testing.assert_array_equal(controller.edit_mask, moved_proxy)
        self.assertEqual(store.revision, moved_revision)

    def test_object_move_store_failure_restores_premutated_proxy(self):
        store = self.make_move_store()
        controller = OverlapEditorController(store, 1)
        source = (np.array([2, 2, 2]), np.array([1, 2, 3]))
        delta = controller.plan_object_move(1, source, (0, 2))
        proxy_before = controller.edit_mask.copy()

        with patch.object(
            store,
            "apply_delta",
            side_effect=MemoryError("forced store failure"),
        ):
            with self.assertRaisesRegex(MemoryError, "forced store"):
                controller.apply_overlap_delta(delta, forward=True)

        np.testing.assert_array_equal(controller.edit_mask, proxy_before)

    def test_class_switch_keeps_array_identities_and_independent_masks(self):
        controller = OverlapEditorController(self.make_store(), 1)
        composite = controller.composite
        edit_mask = controller.edit_mask

        self.assertEqual(edit_mask[2, 2], 1)
        with patch.object(
            controller.store,
            "project",
            side_effect=AssertionError("class switch reprojected composite"),
        ):
            controller.select_class(2)
        self.assertIs(controller.composite, composite)
        self.assertIs(controller.edit_mask, edit_mask)
        self.assertEqual(edit_mask[2, 2], 0)
        self.assertEqual(edit_mask[4, 4], 1)
        self.assertEqual(composite[4, 4], 2)

        with patch.object(
            controller.store,
            "project",
            side_effect=AssertionError("class switch reprojected composite"),
        ):
            controller.select_class(1)
        self.assertEqual(edit_mask[2, 2], 1)
        self.assertEqual(edit_mask[4, 4], 1)
        self.assertEqual(composite[4, 4], 2)

    def test_full_sync_supports_undo_redo_style_array_changes(self):
        store = self.make_store()
        controller = OverlapEditorController(store, 3)
        controller.edit_mask[0, 0] = 1
        controller.edit_mask[7, 8] = 1

        self.assertEqual(controller.full_sync(), 2)
        self.assertTrue(store.select_plane(3)[0, 0])
        self.assertTrue(store.select_plane(3)[7, 8])

        controller.edit_mask.fill(0)
        self.assertEqual(controller.full_sync(), 2)
        self.assertFalse(np.any(store.select_plane(3)))

    def test_membership_addition_changes_only_the_edited_pixel_top(self):
        store = self.make_store()
        controller = OverlapEditorController(store, 1)
        self.assertEqual(store.z_order, (1, 2, 3))

        # Selecting a class and erasing an existing pixel do not reorder it.
        controller.edit_mask[1, 1] = 0
        self.assertEqual(
            controller.process_indices((np.array([1]), np.array([1]))),
            1,
        )
        self.assertEqual(store.z_order, (1, 2, 3))

        unrelated_overlap = int(controller.composite[4, 4])
        # A real 0 -> 1 edit becomes top only at that pixel. It must not
        # globally reorder the class or silently change another overlap.
        controller.edit_mask[0, 0] = 1
        self.assertEqual(
            controller.process_indices((np.array([0]), np.array([0]))),
            1,
        )
        self.assertEqual(store.z_order, (1, 2, 3))
        self.assertEqual(controller.saved_projection()[0, 0], 1)
        self.assertEqual(controller.composite[4, 4], unrelated_overlap)
        projection = controller.saved_projection()
        restored = OverlapStore.from_payload(
            store.to_payload(projection),
            projection,
        )
        self.assertEqual(restored.project()[0, 0], 1)
        self.assertEqual(restored.project()[4, 4], unrelated_overlap)

        # Pure erasure after the addition leaves that order unchanged.
        controller.edit_mask[0, 0] = 0
        controller.process_indices((np.array([0]), np.array([0])))
        self.assertEqual(store.z_order, (1, 2, 3))

    def test_projection_patch_updates_only_requested_rectangle(self):
        controller = OverlapEditorController(self.make_store(), 3)
        original = controller.composite.copy()
        lower = controller.store.select_plane(1)
        lower[0, 0] = True
        controller.store.update_patch(1, (0, 0), lower[:1, :1])

        patch = controller.refresh_projection_patch((0, 0), (1, 1))

        self.assertEqual(patch[0, 0], 1)
        self.assertFalse(patch.flags.writeable)
        np.testing.assert_array_equal(
            controller.composite[1:, :],
            original[1:, :],
        )
        np.testing.assert_array_equal(
            controller.composite[:1, 1:],
            original[:1, 1:],
        )

    def test_delete_hidden_class_leaves_top_projection(self):
        store = self.make_store()
        controller = OverlapEditorController(store, 3)
        before = controller.composite.copy()

        removed = controller.delete_class(1)

        self.assertGreater(removed, 0)
        self.assertNotIn(1, store.class_values)
        self.assertEqual(controller.composite[4, 4], 2)
        self.assertEqual(controller.composite[4, 4], before[4, 4])

    def test_delete_reassignment_ors_into_active_edit_mask(self):
        store = self.make_store()
        controller = OverlapEditorController(store, 3)
        source = store.select_plane(2)

        removed = controller.delete_class(2, 3)

        self.assertEqual(removed, int(np.count_nonzero(source)))
        self.assertNotIn(2, store.class_values)
        np.testing.assert_array_equal(controller.edit_mask != 0, source)
        np.testing.assert_array_equal(store.select_plane(3), source)
        self.assertEqual(controller.composite[4, 4], 3)

    def test_delete_active_selects_replacement(self):
        store = self.make_store()
        controller = OverlapEditorController(store, 2)
        upper = store.select_plane(2)

        controller.delete_class(2, 3)

        self.assertEqual(controller.active_class, 3)
        np.testing.assert_array_equal(controller.edit_mask != 0, upper)
        self.assertEqual(controller.composite[4, 4], 3)

    def test_rejects_nonbinary_and_out_of_bounds_updates(self):
        controller = OverlapEditorController(self.make_store(), 3)

        with self.assertRaises(ValueError):
            controller.process_patch(np.array([[2]], dtype=np.uint8), (0, 0))
        with self.assertRaises(ValueError):
            controller.process_patch(np.ones((2, 2), dtype=np.uint8), (7, 8))
        with self.assertRaises(IndexError):
            controller.process_indices((np.array([8]), np.array([0])))


if __name__ == "__main__":
    unittest.main()
