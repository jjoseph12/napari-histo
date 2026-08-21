import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import patch

import imageio.v3 as iio
import numpy as np
import pandas as pd
from napari.components import ViewerModel
from napari.layers import Labels
from napari.layers.labels._labels_constants import Mode
from qtpy.QtWidgets import QApplication

from napari_histo_label_editor._embedded_annotations import (
    read_embedded_annotations,
)
from napari_histo_label_editor._fast_fill import overlap_fill
from napari_histo_label_editor._fast_polygon import fast_paint_polygon
from napari_histo_label_editor._class_config import read_class_config
from napari_histo_label_editor._overlap_store import OverlapStore
from napari_histo_label_editor._widget import LabelEditorWidget


class FakeSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self.callbacks):
            callback(*args)


class FakeSaveWorker:
    """Worker whose running flag stays false until after the UI event returns."""

    def __init__(self):
        self.returned = FakeSignal()
        self.errored = FakeSignal()
        self.finished = FakeSignal()
        self.is_running = False
        self.start_count = 0

    def start(self):
        self.start_count += 1


class LabelFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.features = LabelEditorWidget._label_features(
            {3: "Tumor", 0: "background", 12: "Stroma"}
        )

    def test_label_features_include_label_values_and_names(self):
        expected = pd.DataFrame(
            {
                "index": [0, 3, 12],
                "Label": ["0 — background", "3 — Tumor", "12 — Stroma"],
            }
        )
        pd.testing.assert_frame_equal(self.features, expected)

    def test_napari_tooltip_shows_hovered_label(self):
        layer = Labels(
            np.array([[0, 3], [12, 0]], dtype=np.int32),
            features=self.features,
        )

        self.assertEqual(layer._get_tooltip_text((0, 1)), "Label: 3 — Tumor")


class LargeLabelPerformanceTest(unittest.TestCase):
    def test_int32_semantic_labels_are_compacted_without_changing_values(self):
        labels = np.array([[0, 1], [12, 21]], dtype=np.int32)

        compact = LabelEditorWidget._compact_labels(
            labels,
            {0: "background", 1: "Tumor", 12: "Stroma", 21: "Other"},
        )

        self.assertEqual(compact.dtype, np.dtype(np.uint8))
        np.testing.assert_array_equal(compact, labels)

    def test_dtype_also_accommodates_class_values_not_yet_in_mask(self):
        labels = np.zeros((2, 2), dtype=np.int32)

        compact = LabelEditorWidget._compact_labels(
            labels,
            {0: "background", 300: "Future class"},
        )

        self.assertEqual(compact.dtype, np.dtype(np.uint16))

    def test_plugin_undo_uses_napari_changed_pixel_history(self):
        layer = Labels(np.zeros((4, 4), dtype=np.uint8))
        indices = (np.array([1]), np.array([2]))
        layer.data_setitem(indices, 3)

        self.assertTrue(LabelEditorWidget._undo_labels_layer(layer))
        self.assertEqual(layer.data[1, 2], 0)
        self.assertFalse(LabelEditorWidget._undo_labels_layer(layer))

    def test_new_class_values_only_widen_dtypes_when_required(self):
        class_map = {0: "background", 300: "Large class"}

        self.assertEqual(
            LabelEditorWidget._promoted_dtype_for_classes(
                np.dtype(np.uint8), class_map
            ),
            np.dtype(np.uint16),
        )
        self.assertEqual(
            LabelEditorWidget._promoted_dtype_for_classes(
                np.dtype(np.int32), class_map
            ),
            np.dtype(np.int32),
        )


class LoadSaveIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_for_save(self, widget, timeout=5):
        deadline = monotonic() + timeout
        while widget._save_worker is not None:
            self.app.processEvents()
            if monotonic() >= deadline:
                self.fail("Background label save did not finish")
            sleep(0.005)

    @staticmethod
    def load_small_project(
        root,
        *,
        labels_dtype=np.uint8,
        labels_suffix=".tif",
    ):
        image_path = root / "image.png"
        labels_path = root / f"labels{labels_suffix}"
        mapping_path = root / "mapping.csv"
        iio.imwrite(image_path, np.zeros((8, 10, 3), dtype=np.uint8))
        labels = np.zeros((8, 10), dtype=labels_dtype)
        labels[1, 1] = 1
        iio.imwrite(labels_path, labels)
        pd.DataFrame(
            {"value": [0, 1], "name": ["background", "Tumor"]}
        ).to_csv(mapping_path, index=False)

        viewer = ViewerModel()
        widget = LabelEditorWidget(viewer)
        widget.image_line.setText(str(image_path))
        widget.label_line.setText(str(labels_path))
        widget.mapping_line.setText(str(mapping_path))
        widget.load_data()
        return viewer, widget, labels_path, mapping_path

    def test_large_mask_optimizations_are_automatic_and_lossless(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            labels_path = root / "labels.tif"
            mapping_path = root / "mapping.csv"

            iio.imwrite(
                image_path,
                np.zeros((16, 20, 3), dtype=np.uint8),
            )
            original = np.array([[0, 1] * 10] * 16, dtype=np.int32)
            iio.imwrite(labels_path, original)
            pd.DataFrame(
                {
                    "value": [0, 1],
                    "name": ["background", "tumor"],
                }
            ).to_csv(mapping_path, index=False)

            viewer = ViewerModel()
            widget = LabelEditorWidget(viewer)
            widget.image_line.setText(str(image_path))
            widget.label_line.setText(str(labels_path))
            widget.mapping_line.setText(str(mapping_path))
            widget.load_data()

            self.assertEqual(
                [layer.name for layer in viewer.layers],
                [
                    "histology",
                    "labels — all classes",
                    "active class — 1: tumor",
                ],
            )
            self.assertFalse(viewer.layers[0].multiscale)
            self.assertEqual(viewer.layers[0].interpolation2d, "linear")
            self.assertEqual(widget.labels_layer.data.dtype, np.dtype(np.uint8))
            self.assertEqual(widget.composite_layer.data.dtype, np.dtype(np.uint8))
            self.assertEqual(widget._labels_output_dtype, np.dtype(np.int32))
            self.assertEqual(widget.labels_layer.n_edit_dimensions, 2)
            self.assertTrue(widget.labels_layer.contiguous)
            self.assertEqual(widget.labels_layer._undo_history.maxlen, 20)
            self.assertFalse(widget.composite_layer.data.flags.writeable)
            self.assertFalse(widget.composite_layer.editable)
            self.assertIs(widget.labels_layer.fill.__func__, overlap_fill)
            self.assertIs(
                widget.labels_layer.paint_polygon.__func__,
                fast_paint_polygon,
            )

            widget.labels_layer.data[0, 0] = 1
            widget.save_labels()
            self.wait_for_save(widget)
            saved = iio.imread(labels_path)

            self.assertEqual(saved.shape, original.shape)
            self.assertEqual(saved.dtype, original.dtype)
            self.assertEqual(saved[0, 0], 1)

    def test_save_preserves_sparse_undo_history(self):
        with TemporaryDirectory() as tmp:
            _, widget, labels_path, _ = self.load_small_project(Path(tmp))
            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                1,
            )
            native_history_size = len(widget.labels_layer._undo_history)
            top_history_size = len(
                widget.labels_layer._napari_histo_edit_tracker.undo_items
            )

            widget.save_labels()
            self.wait_for_save(widget)

            self.assertEqual(
                len(widget.labels_layer._undo_history),
                native_history_size,
            )
            self.assertEqual(
                len(widget.labels_layer._napari_histo_edit_tracker.undo_items),
                top_history_size,
            )
            self.assertEqual(iio.imread(labels_path)[2, 3], 1)
            widget.undo()
            self.assertEqual(widget.labels_layer.data[2, 3], 0)
            self.assertEqual(widget.composite_layer.data[2, 3], 0)

    def test_overlap_survives_save_and_reload_in_the_same_label_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            viewer, widget, labels_path, mapping_path = self.load_small_project(
                root
            )
            # Add class 2 beneath the active class at one pixel, then paint
            # class 1 over it through napari's normal partial-update path.
            widget._upsert_class(2, "Stroma", "#123456")
            widget._select_label(2)
            widget.labels_layer.data_setitem(
                (np.array([3]), np.array([4])),
                1,
            )
            widget._select_label(1)
            widget.labels_layer.data_setitem(
                (np.array([3]), np.array([4])),
                1,
            )

            self.assertEqual(
                widget.overlap_store.memberships_at(3, 4),
                (2, 1),
            )
            widget.save_labels()
            self.wait_for_save(widget)

            projection = iio.imread(labels_path)
            self.assertEqual(projection[3, 4], 1)
            payload = read_embedded_annotations(labels_path)
            self.assertIsNotNone(payload)
            restored = OverlapStore.from_payload(payload, projection)
            self.assertEqual(restored.memberships_at(3, 4), (2, 1))
            self.assertEqual(list(labels_path.parent.glob("*.overlap*")), [])

            # A fresh widget recovers both memberships from that same TIFF.
            reloaded_viewer = ViewerModel()
            reloaded = LabelEditorWidget(reloaded_viewer)
            reloaded.image_line.setText(str(widget.image_path))
            reloaded.label_line.setText(str(labels_path))
            reloaded.mapping_line.setText(str(mapping_path))
            reloaded.load_data()
            self.assertEqual(
                reloaded.overlap_store.memberships_at(3, 4),
                (2, 1),
            )

    def test_overlap_edit_uploads_only_the_changed_composite_patch(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            emitted_updates = []
            emitted_offsets = []

            def record_event(event):
                emitted_updates.append(np.array(event.data, copy=True))
                emitted_offsets.append(tuple(event.offset))

            widget.composite_layer.events.labels_update.connect(record_event)

            with patch.object(widget.composite_layer, "refresh") as full_refresh:
                widget.labels_layer.data_setitem(
                    (np.array([2]), np.array([3])),
                    1,
                )

            self.assertEqual(emitted_offsets, [(2, 3)])
            full_refresh.assert_not_called()
            self.assertEqual(widget.composite_layer.data[2, 3], 1)
            updated_slice = (slice(2, 3), slice(3, 4))
            expected_texture = widget.composite_layer._raw_to_displayed(
                widget.composite_layer._slice.image.raw,
                data_slice=updated_slice,
            )
            np.testing.assert_array_equal(
                widget.composite_layer._slice.image.view[updated_slice],
                expected_texture,
            )
            self.assertTrue(emitted_updates)
            np.testing.assert_array_equal(
                emitted_updates[-1],
                expected_texture,
            )

    def test_class_switch_does_not_reproject_or_refresh_composite(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            widget._upsert_class(2, "Stroma", "#123456")
            composite_identity = id(widget.overlap_editor.composite)
            previous_preview_color = widget.labels_layer._selected_color.copy()

            with patch.object(
                widget.overlap_editor,
                "refresh_projection",
                wraps=widget.overlap_editor.refresh_projection,
            ) as reproject, patch.object(
                widget.composite_layer,
                "refresh",
            ) as full_refresh, patch.object(
                widget.labels_layer,
                "refresh",
            ) as hidden_refresh:
                widget._select_label(1)

            reproject.assert_not_called()
            full_refresh.assert_not_called()
            hidden_refresh.assert_not_called()
            self.assertEqual(id(widget.overlap_editor.composite), composite_identity)
            self.assertEqual(widget.overlap_editor.active_class, 1)
            self.assertFalse(
                np.array_equal(
                    widget.labels_layer._selected_color,
                    previous_preview_color,
                )
            )
            np.testing.assert_allclose(
                widget.labels_layer._selected_color,
                widget._label_rgba(1),
            )
            np.testing.assert_allclose(
                widget.labels_layer.get_color(1),
                widget._label_rgba(1),
                atol=1 / 255,
            )

    def test_failed_second_load_keeps_original_layer_and_save_destination(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_a = root / "project-a"
            project_b = root / "project-b"
            project_a.mkdir()
            project_b.mkdir()
            viewer, widget, labels_a, _ = self.load_small_project(project_a)
            original_layer = widget.labels_layer
            original_labels_a = iio.imread(labels_a).copy()

            image_b = project_b / "image.png"
            labels_b = project_b / "labels.tif"
            mapping_b = project_b / "mapping.csv"
            iio.imwrite(image_b, np.zeros((8, 10, 3), dtype=np.uint8))
            iio.imwrite(labels_b, np.full((7, 10), 2, dtype=np.uint8))
            pd.DataFrame(
                {"value": [0, 2], "name": ["background", "Stroma"]}
            ).to_csv(mapping_b, index=False)
            original_labels_b = iio.imread(labels_b).copy()

            widget.image_line.setText(str(image_b))
            widget.label_line.setText(str(labels_b))
            widget.mapping_line.setText(str(mapping_b))

            with self.assertRaisesRegex(ValueError, "differ"):
                widget.load_data()

            self.assertEqual(widget.labels_path, labels_a.resolve())
            self.assertIs(widget.labels_layer, original_layer)
            self.assertTrue(any(layer is original_layer for layer in viewer.layers))
            self.assertTrue(widget.save_btn.isEnabled())
            self.assertEqual(widget.save_btn.text(), "Save [s]")
            self.assertEqual(
                widget.save_destination_line.text(),
                str(labels_a.resolve()),
            )

            original_layer.data[2, 2] = 1
            widget.save_labels()
            self.wait_for_save(widget)

            expected_labels_a = original_labels_a.copy()
            expected_labels_a[2, 2] = 1
            np.testing.assert_array_equal(iio.imread(labels_a), expected_labels_a)
            np.testing.assert_array_equal(iio.imread(labels_b), original_labels_b)

    def test_png_mapping_above_uint16_fails_before_replacing_active_project(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_a = root / "project-a"
            project_b = root / "project-b"
            project_a.mkdir()
            project_b.mkdir()
            viewer, widget, labels_a, mapping_a = self.load_small_project(
                project_a
            )
            previous_layers = list(viewer.layers)
            previous_store = widget.overlap_store
            previous_editor = widget.overlap_editor
            previous_class_map = dict(widget.class_map)
            previous_labels = labels_a.read_bytes()
            previous_mapping = mapping_a.read_bytes()

            image_b = project_b / "image.png"
            labels_b = project_b / "labels.png"
            mapping_b = project_b / "mapping.csv"
            iio.imwrite(image_b, np.zeros((8, 10, 3), dtype=np.uint8))
            iio.imwrite(labels_b, np.zeros((8, 10), dtype=np.uint8))
            pd.DataFrame(
                {
                    "value": [0, 70000],
                    "name": ["background", "Too large"],
                }
            ).to_csv(mapping_b, index=False)

            widget.image_line.setText(str(image_b))
            widget.label_line.setText(str(labels_b))
            widget.mapping_line.setText(str(mapping_b))
            with self.assertRaisesRegex(
                ValueError,
                "PNG supports class IDs up to 65535; use TIFF",
            ):
                widget.load_data()

            self.assertEqual(list(viewer.layers), previous_layers)
            self.assertIs(widget.overlap_store, previous_store)
            self.assertIs(widget.overlap_editor, previous_editor)
            self.assertEqual(widget.labels_path, labels_a.resolve())
            self.assertEqual(widget.mapping_path, mapping_a.resolve())
            self.assertEqual(widget.class_map, previous_class_map)
            self.assertEqual(labels_a.read_bytes(), previous_labels)
            self.assertEqual(mapping_a.read_bytes(), previous_mapping)

    def test_moved_loaded_destination_is_not_silently_recreated(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)
            moved_labels_path = root / "labels-moved.tif"
            original = iio.imread(labels_path).copy()
            labels_path.rename(moved_labels_path)
            widget.labels_layer.data[2, 3] = 1

            with patch(
                "napari_histo_label_editor._widget.QFileDialog.getSaveFileName",
                return_value=("", ""),
            ) as save_as_dialog, patch(
                "napari_histo_label_editor._widget.create_worker"
            ) as create_worker:
                widget.save_labels()

            save_as_dialog.assert_called_once()
            create_worker.assert_not_called()
            self.assertFalse(labels_path.exists())
            self.assertFalse(widget.labels_path.exists())
            self.assertTrue(moved_labels_path.is_file())
            np.testing.assert_array_equal(iio.imread(moved_labels_path), original)
            self.assertTrue(widget.save_btn.isEnabled())
            self.assertEqual(widget.save_btn.text(), "Save As… [s]")

    def test_save_as_rescues_all_overlaps_and_becomes_next_save_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)
            widget._upsert_class(2, "Stroma", "#123456")
            widget._select_label(2)
            widget.labels_layer.data_setitem(
                (np.array([1]), np.array([1])),
                1,
            )
            self.assertEqual(
                widget.overlap_store.memberships_at(1, 1),
                (1, 2),
            )
            self.assertTrue(widget.save_btn.isEnabled())

            moved_path = root / "original-moved.tif"
            labels_path.rename(moved_path)
            moved_bytes = moved_path.read_bytes()
            rescued_path = root / "rescued.tif"
            with patch(
                "napari_histo_label_editor._widget.QFileDialog.getSaveFileName",
                return_value=(str(rescued_path), ""),
            ) as save_as_dialog:
                widget.save_labels()
                self.wait_for_save(widget)

            save_as_dialog.assert_called_once()
            self.assertFalse(labels_path.exists())
            self.assertEqual(moved_path.read_bytes(), moved_bytes)
            self.assertEqual(widget.labels_path, rescued_path.resolve())
            self.assertEqual(widget.label_line.text(), str(rescued_path.resolve()))
            self.assertEqual(
                widget.save_destination_line.text(),
                str(rescued_path.resolve()),
            )
            projection = iio.imread(rescued_path)
            payload = read_embedded_annotations(rescued_path)
            self.assertIsNotNone(payload)
            restored = OverlapStore.from_payload(payload, projection=projection)
            self.assertEqual(restored.memberships_at(1, 1), (1, 2))

            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                1,
            )
            self.assertTrue(widget.save_btn.isEnabled())
            with patch(
                "napari_histo_label_editor._widget.QFileDialog.getSaveFileName"
            ) as save_as_dialog:
                widget.save_labels()
                self.wait_for_save(widget)
            save_as_dialog.assert_not_called()
            self.assertEqual(iio.imread(rescued_path)[2, 3], 2)

    def test_save_as_refuses_project_inputs_and_symlink_aliases(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)
            image_bytes = widget.image_path.read_bytes()
            mapping_bytes = mapping_path.read_bytes()
            mapping_alias = root / "mapping-alias.tif"
            mapping_alias.symlink_to(mapping_path)

            for forbidden in (widget.image_path, mapping_alias):
                with self.subTest(forbidden=forbidden), patch(
                    "napari_histo_label_editor._widget.QFileDialog.getSaveFileName",
                    return_value=(str(forbidden), ""),
                ), self.assertRaisesRegex(ValueError, "cannot overwrite"):
                    widget._choose_save_as_destination()

            self.assertEqual(widget.image_path.read_bytes(), image_bytes)
            self.assertEqual(mapping_path.read_bytes(), mapping_bytes)

            replacement = np.full((8, 10), 7, dtype=np.uint8)
            replacement_path = root / "replacement.tif"
            iio.imwrite(replacement_path, replacement)
            os.replace(replacement_path, labels_path)
            compromised_alias = root / "compromised-alias.tif"
            compromised_alias.symlink_to(labels_path)
            with patch(
                "napari_histo_label_editor._widget.QFileDialog.getSaveFileName",
                return_value=(str(compromised_alias), ""),
            ), self.assertRaisesRegex(ValueError, "missing or changed"):
                widget._choose_save_as_destination()
            np.testing.assert_array_equal(iio.imread(labels_path), replacement)

    def test_add_labels_post_insert_failure_restores_exact_previous_project(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_a = root / "project-a"
            project_b = root / "project-b"
            project_a.mkdir()
            project_b.mkdir()
            viewer, widget, labels_a, _ = self.load_small_project(project_a)

            previous_layers = list(viewer.layers)
            previous_labels_layer = widget.labels_layer
            previous_image_path = widget.image_path
            previous_labels_path = widget.labels_path
            previous_mapping_path = widget.mapping_path
            previous_class_map = widget.class_map
            previous_class_colors = widget.class_colors
            previous_output_dtype = widget._labels_output_dtype
            previous_labels_identity = widget._labels_destination_identity
            previous_mapping_identity = widget._mapping_destination_identity
            previous_save_target = widget.save_destination_line.text()

            image_b = project_b / "image.png"
            labels_b = project_b / "labels.tif"
            mapping_b = project_b / "mapping.csv"
            iio.imwrite(image_b, np.ones((8, 10, 3), dtype=np.uint8))
            labels = np.zeros((8, 10), dtype=np.uint8)
            labels[4, 5] = 2
            iio.imwrite(labels_b, labels)
            original_labels_b = labels.copy()
            pd.DataFrame(
                {
                    "value": [0, 2],
                    "name": ["background", "Stroma"],
                    "color": ["#000000", "#123456"],
                }
            ).to_csv(mapping_b, index=False)
            widget.image_line.setText(str(image_b))
            widget.label_line.setText(str(labels_b))
            widget.mapping_line.setText(str(mapping_b))
            inserted_layers = []
            real_add_labels = type(viewer).add_labels

            def add_labels_then_raise(active_viewer, *args, **kwargs):
                inserted_layer = real_add_labels(
                    active_viewer,
                    *args,
                    **kwargs,
                )
                inserted_layers.append(inserted_layer)
                raise RuntimeError("simulated post-insert add_labels failure")

            with patch.object(
                type(viewer),
                "add_labels",
                new=add_labels_then_raise,
            ), self.assertRaisesRegex(RuntimeError, "simulated"):
                widget.load_data()

            self.assertEqual(len(inserted_layers), 1)
            self.assertEqual(len(viewer.layers), len(previous_layers))
            for actual, expected in zip(viewer.layers, previous_layers):
                self.assertIs(actual, expected)
            self.assertFalse(
                any(
                    layer is inserted_layers[0]
                    for layer in viewer.layers
                )
            )
            self.assertIs(widget.labels_layer, previous_labels_layer)
            self.assertIs(widget.image_path, previous_image_path)
            self.assertIs(widget.labels_path, previous_labels_path)
            self.assertIs(widget.mapping_path, previous_mapping_path)
            self.assertIs(widget.class_map, previous_class_map)
            self.assertIs(widget.class_colors, previous_class_colors)
            self.assertEqual(widget._labels_output_dtype, previous_output_dtype)
            self.assertIs(
                widget._labels_destination_identity,
                previous_labels_identity,
            )
            self.assertIs(
                widget._mapping_destination_identity,
                previous_mapping_identity,
            )
            self.assertEqual(
                widget.save_destination_line.text(),
                previous_save_target,
            )
            self.assertTrue(widget.save_btn.isEnabled())
            self.assertEqual(widget.save_btn.text(), "Save [s]")

            previous_labels_layer.data[2, 3] = 1
            widget.save_labels()
            self.wait_for_save(widget)

            self.assertEqual(iio.imread(labels_a)[2, 3], 1)
            np.testing.assert_array_equal(iio.imread(labels_b), original_labels_b)

    def test_replaced_regular_label_file_is_not_overwritten(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)
            replacement_path = root / "replacement.tif"
            replacement = np.full((8, 10), 7, dtype=np.uint8)
            iio.imwrite(replacement_path, replacement)
            os.replace(replacement_path, labels_path)
            widget.labels_layer.data[2, 3] = 1

            with patch(
                "napari_histo_label_editor._widget.QFileDialog.getSaveFileName",
                return_value=("", ""),
            ) as save_as_dialog, patch(
                "napari_histo_label_editor._widget.create_worker"
            ) as create_worker:
                widget.save_labels()

            save_as_dialog.assert_called_once()
            create_worker.assert_not_called()
            np.testing.assert_array_equal(iio.imread(labels_path), replacement)
            self.assertTrue(widget.save_btn.isEnabled())
            self.assertEqual(widget.save_btn.text(), "Save As… [s]")

    def test_mapping_replaced_by_symlink_cannot_update_unrelated_csv(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, _, mapping_path = self.load_small_project(root)
            victim_path = root / "unrelated-victim.csv"
            pd.DataFrame(
                {"value": [0, 99], "name": ["background", "Do not edit"]}
            ).to_csv(victim_path, index=False)
            victim_before = victim_path.read_bytes()
            class_map_before = dict(widget.class_map)
            class_colors_before = dict(widget.class_colors)

            mapping_path.unlink()
            mapping_path.symlink_to(victim_path)

            with self.assertRaisesRegex(ValueError, "missing or changed"):
                widget._upsert_class(2, "Stroma", "#123456")

            self.assertTrue(mapping_path.is_symlink())
            self.assertEqual(victim_path.read_bytes(), victim_before)
            self.assertEqual(widget.class_map, class_map_before)
            self.assertEqual(widget.class_colors, class_colors_before)

    def test_two_consecutive_real_saves_refresh_destination_identity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)

            widget.labels_layer.data[2, 3] = 1
            widget.save_labels()
            self.wait_for_save(widget)

            self.assertEqual(iio.imread(labels_path)[2, 3], 1)
            self.assertTrue(widget.save_btn.isEnabled())

            widget.labels_layer.data[4, 5] = 1
            widget.save_labels()
            self.wait_for_save(widget)

            saved = iio.imread(labels_path)
            self.assertEqual(saved[2, 3], 1)
            self.assertEqual(saved[4, 5], 1)
            self.assertTrue(widget.save_btn.isEnabled())

    def test_staged_label_path_keeps_save_locked_to_loaded_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)
            other_labels_path = root / "other-labels.tif"
            other_labels = np.zeros((8, 10), dtype=np.uint8)
            other_labels[3, 4] = 1
            iio.imwrite(other_labels_path, other_labels)
            original_other_labels = other_labels.copy()

            widget.label_line.setText(str(other_labels_path))

            self.assertTrue(widget.save_btn.isEnabled())
            self.assertEqual(widget.save_btn.text(), "Save [s]")
            self.assertEqual(widget.labels_path, labels_path.resolve())
            self.assertEqual(
                widget.save_destination_line.text(),
                str(labels_path.resolve()),
            )

            widget.labels_layer.data[2, 3] = 1
            widget.save_labels()
            self.wait_for_save(widget)

            self.assertEqual(iio.imread(labels_path)[2, 3], 1)
            np.testing.assert_array_equal(
                iio.imread(other_labels_path),
                original_other_labels,
            )
            self.assertEqual(widget.save_btn.text(), "Save [s]")

            widget.load_data()

            self.assertEqual(widget.labels_path, other_labels_path.resolve())
            self.assertTrue(widget.save_btn.isEnabled())
            self.assertEqual(widget.save_btn.text(), "Save [s]")

    def test_rapid_double_save_creates_only_one_worker(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            workers = []

            def make_worker(*args, **kwargs):
                del args, kwargs
                worker = FakeSaveWorker()
                workers.append(worker)
                return worker

            with patch(
                "napari_histo_label_editor._widget.create_worker",
                side_effect=make_worker,
            ):
                widget.save_labels()
                widget.save_labels()

            self.assertEqual(len(workers), 1)
            self.assertEqual(workers[0].start_count, 1)
            self.assertIs(widget._save_worker, workers[0])
            workers[0].finished.emit()

    def test_programmatic_edit_cannot_bypass_active_save_lock_or_break_undo(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            widget._upsert_class(2, "Stroma", "#123456")
            widget._select_label(1)
            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                1,
            )
            data_before = widget.labels_layer.data.copy()
            projection_before = widget.overlap_store.project()
            active_before = widget.overlap_editor.active_class
            native_history_before = len(widget.labels_layer._undo_history)
            native_redo_before = len(widget.labels_layer._redo_history)
            top_history_before = len(
                widget.labels_layer._napari_histo_edit_tracker.undo_items
            )
            top_redo_before = len(
                widget.labels_layer._napari_histo_edit_tracker.redo_items
            )
            worker = FakeSaveWorker()

            with patch(
                "napari_histo_label_editor._widget.create_worker",
                return_value=worker,
            ):
                widget.save_labels()

            # Simulate a caller overriding the UI's PAN_ZOOM lock while the
            # worker still owns the packed store.
            widget.labels_layer.mode = Mode.PAINT
            widget.labels_layer.data_setitem(
                (np.array([4]), np.array([5])),
                1,
            )
            # Native Cmd-Z/Cmd-Shift-Z dispatch directly to these layer
            # methods, so both must obey the same worker ownership lock.
            widget.labels_layer.undo()
            widget.labels_layer.redo()
            widget._select_label(2)
            event = SimpleNamespace(
                position=(0, 0),
                view_direction=None,
                dims_displayed=(0, 1),
            )
            with patch.object(
                widget.composite_layer,
                "get_value",
                return_value=2,
            ):
                widget.labels_layer._drag_modes[Mode.PICK](
                    widget.labels_layer,
                    event,
                )

            np.testing.assert_array_equal(widget.labels_layer.data, data_before)
            np.testing.assert_array_equal(
                widget.overlap_store.project(),
                projection_before,
            )
            self.assertEqual(widget.overlap_editor.active_class, active_before)
            self.assertEqual(
                len(widget.labels_layer._undo_history),
                native_history_before,
            )
            self.assertEqual(
                len(widget.labels_layer._redo_history),
                native_redo_before,
            )
            self.assertEqual(
                len(widget.labels_layer._napari_histo_edit_tracker.undo_items),
                top_history_before,
            )
            self.assertEqual(
                len(widget.labels_layer._napari_histo_edit_tracker.redo_items),
                top_redo_before,
            )
            self.assertIn("save to finish", widget.viewer.status)

            worker.finished.emit()
            widget.undo()
            self.assertEqual(widget.labels_layer.data[2, 3], 0)
            self.assertEqual(widget.composite_layer.data[2, 3], 0)
            widget.labels_layer.redo()
            self.assertEqual(widget.labels_layer.data[2, 3], 1)
            self.assertEqual(widget.composite_layer.data[2, 3], 1)

    def test_removing_owned_labels_layer_disables_and_refuses_save(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            viewer, widget, labels_path, _ = self.load_small_project(root)
            original = iio.imread(labels_path).copy()
            removed_layer = widget.labels_layer

            viewer.layers.remove(removed_layer)

            self.assertFalse(widget.save_btn.isEnabled())
            with patch(
                "napari_histo_label_editor._widget.create_worker"
            ) as create_worker:
                widget.save_labels()
            create_worker.assert_not_called()
            np.testing.assert_array_equal(iio.imread(labels_path), original)

    def test_load_is_blocked_while_a_save_worker_is_owned(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_a = root / "project-a"
            project_b = root / "project-b"
            project_a.mkdir()
            project_b.mkdir()
            viewer, widget, labels_a, _ = self.load_small_project(project_a)
            original_layer = widget.labels_layer
            _, other_widget, labels_b, mapping_b = self.load_small_project(
                project_b
            )
            image_b = other_widget.image_path
            other_widget.deleteLater()
            worker = FakeSaveWorker()

            with patch(
                "napari_histo_label_editor._widget.create_worker",
                return_value=worker,
            ):
                widget.save_labels()

            self.assertFalse(widget.load_btn.isEnabled())
            widget.image_line.setText(str(image_b))
            widget.label_line.setText(str(labels_b))
            widget.mapping_line.setText(str(mapping_b))
            widget.load_data()

            self.assertEqual(widget.labels_path, labels_a.resolve())
            self.assertIs(widget.labels_layer, original_layer)
            self.assertTrue(any(layer is original_layer for layer in viewer.layers))
            self.assertIs(widget._save_worker, worker)

            worker.finished.emit()
            self.assertTrue(widget.load_btn.isEnabled())

    def test_relative_paths_are_canonicalized_before_cwd_changes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_a = root / "project-a"
            project_b = root / "project-b"
            project_a.mkdir()
            project_b.mkdir()

            for project in (project_a, project_b):
                iio.imwrite(
                    project / "image.png",
                    np.zeros((8, 10, 3), dtype=np.uint8),
                )
                iio.imwrite(
                    project / "labels.tif",
                    np.zeros((8, 10), dtype=np.uint8),
                )
                pd.DataFrame(
                    {"value": [0, 1], "name": ["background", "Tumor"]}
                ).to_csv(project / "mapping.csv", index=False)

            previous_cwd = Path.cwd()
            try:
                os.chdir(project_a)
                viewer = ViewerModel()
                widget = LabelEditorWidget(viewer)
                widget.image_line.setText("image.png")
                widget.label_line.setText("labels.tif")
                widget.mapping_line.setText("mapping.csv")
                widget.load_data()

                self.assertEqual(
                    widget.labels_path,
                    (project_a / "labels.tif").resolve(),
                )
                self.assertEqual(
                    widget.label_line.text(),
                    str((project_a / "labels.tif").resolve()),
                )

                os.chdir(project_b)
                widget.labels_layer.data[2, 3] = 1
                widget.save_labels()
                self.wait_for_save(widget)
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(iio.imread(project_a / "labels.tif")[2, 3], 1)
            self.assertEqual(iio.imread(project_b / "labels.tif")[2, 3], 0)

    def test_oversized_rgb_uses_filtered_multiscale_image_data(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            labels_path = root / "labels.tif"
            mapping_path = root / "mapping.csv"
            image = np.zeros((16, 20, 3), dtype=np.uint8)
            coarse = np.zeros((8, 10, 3), dtype=np.uint8)
            iio.imwrite(image_path, image)
            iio.imwrite(labels_path, np.zeros((16, 20), dtype=np.uint8))
            pd.DataFrame(
                {"value": [0, 1], "name": ["background", "Tumor"]}
            ).to_csv(mapping_path, index=False)

            viewer = ViewerModel()
            widget = LabelEditorWidget(viewer)
            widget.image_line.setText(str(image_path))
            widget.label_line.setText(str(labels_path))
            widget.mapping_line.setText(str(mapping_path))

            with patch(
                "napari_histo_label_editor._widget.build_image_pyramid",
                return_value=[image, coarse],
            ):
                widget.load_data()

            histology = viewer.layers[0]
            self.assertTrue(histology.multiscale)
            self.assertEqual(histology.interpolation2d, "linear")
            self.assertEqual(
                [level.shape for level in histology.data],
                [(16, 20, 3), (8, 10, 3)],
            )

    def test_layer_changes_restore_the_plugin_polygon_preview(self):
        viewer = ViewerModel()
        widget = LabelEditorWidget(viewer)
        widget.labels_layer = viewer.add_labels(
            np.zeros((8, 10), dtype=np.uint8)
        )

        with patch(
            "napari_histo_label_editor._widget.enable_fast_rendering"
        ) as enable:
            extra = viewer.add_image(np.zeros((4, 5), dtype=np.uint8))
            enable.assert_called_once_with(viewer, widget.labels_layer)

            enable.reset_mock()
            viewer.layers.remove(extra)
            enable.assert_called_once_with(viewer, widget.labels_layer)

    def test_add_and_edit_class_updates_csv_layer_and_color_immediately(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            viewer, widget, _, mapping_path = self.load_small_project(root)

            widget._upsert_class(2, "Stroma", "#123456")

            self.assertEqual(widget.class_map[2], "Stroma")
            self.assertEqual(widget.class_colors[2], "#123456")
            self.assertEqual(widget.overlap_editor.active_class, 2)
            self.assertEqual(widget.labels_layer.selected_label, 1)
            self.assertIs(viewer.layers.selection.active, widget.labels_layer)
            self.assertFalse(widget.labels_layer.preserve_labels)
            np.testing.assert_allclose(
                widget.composite_layer.get_color(2),
                np.array([0x12, 0x34, 0x56, 0xFF]) / 255,
                atol=1 / 255,
            )

            widget.labels_layer.data_setitem(
                (np.array([1]), np.array([1])),
                1,
            )
            self.assertEqual(
                widget.labels_layer._get_tooltip_text((1, 1)),
                "Top: 2 — Stroma | Memberships: 1 — Tumor, 2 — Stroma",
            )
            self.assertEqual(
                widget.labels_layer._get_tooltip_text(
                    (1, 1),
                    view_direction=np.array([0.0, 0.0]),
                    dims_displayed=[0, 1],
                    world=True,
                ),
                "Top: 2 — Stroma | Memberships: 1 — Tumor, 2 — Stroma",
            )

            widget._upsert_class(2, "Fibrosis", "red")
            class_map, class_colors = read_class_config(mapping_path)
            rows = pd.read_csv(mapping_path)

            self.assertEqual(class_map[2], "Fibrosis")
            self.assertEqual(class_colors[2], "#ff0000")
            self.assertEqual((rows["value"] == 2).sum(), 1)
            self.assertEqual(
                widget.labels_layer._get_tooltip_text((1, 1)),
                "Top: 2 — Fibrosis | Memberships: 1 — Tumor, 2 — Fibrosis",
            )
            np.testing.assert_allclose(
                widget.composite_layer.get_color(2),
                np.array([1.0, 0.0, 0.0, 1.0]),
            )

    def test_semantic_tooltip_reports_top_overlap_background_and_high_id(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            widget._upsert_class(2, "Stroma", "#123456")
            widget.labels_layer.data_setitem(
                (np.array([1]), np.array([1])),
                1,
            )
            widget._select_label(1)

            self.assertEqual(widget.overlap_editor.active_class, 1)
            self.assertEqual(widget.composite_layer.data[1, 1], 2)
            self.assertEqual(
                widget.labels_layer._get_tooltip_text((1, 1)),
                "Top: 2 — Stroma | Memberships: 1 — Tumor, 2 — Stroma",
            )
            self.assertEqual(
                widget.labels_layer._get_tooltip_text((0, 0)),
                "Label: 0 — background",
            )
            self.assertEqual(
                widget.labels_layer._get_tooltip_text((-1, -1)),
                "",
            )

            widget._upsert_class(300, "Review", "#abcdef")
            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                1,
            )
            self.assertEqual(
                widget.labels_layer._get_tooltip_text((2, 3)),
                "Label: 300 — Review",
            )

    def test_plugin_opacity_slider_controls_only_visible_composite(self):
        viewer = ViewerModel()
        empty_widget = LabelEditorWidget(viewer)
        self.assertFalse(empty_widget.overlay_opacity_slider.isEnabled())

        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            self.assertTrue(widget.overlay_opacity_slider.isEnabled())
            self.assertEqual(widget.overlay_opacity_slider.value(), 45)
            self.assertAlmostEqual(widget.composite_layer.opacity, 0.45)
            self.assertEqual(widget.labels_layer.opacity, 0.0)

            widget.overlay_opacity_slider.setValue(67)
            self.assertEqual(widget.overlay_opacity_value.text(), "67%")
            self.assertAlmostEqual(widget.composite_layer.opacity, 0.67)
            self.assertEqual(widget.labels_layer.opacity, 0.0)

            widget.labels_layer.opacity = 0.8
            self.assertEqual(widget.labels_layer.opacity, 0.0)

            widget._save_worker = object()
            widget._update_project_controls()
            self.assertFalse(widget.overlay_opacity_slider.isEnabled())

    def test_repaint_hidden_memberships_is_local_and_undo_redo_exact(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)

            # Build class A broadly, then class B above it in four disjoint
            # regions used by Paint, Polygon, Fill, and an unrelated control.
            a_rows, a_columns = np.indices((6, 8))
            widget.labels_layer.data_setitem(
                (a_rows.reshape(-1) + 1, a_columns.reshape(-1) + 1),
                1,
            )
            widget._upsert_class(2, "B", "#123456")
            b_coordinates = np.array(
                [
                    [1, 1],
                    [1, 4],
                    [1, 5],
                    [2, 4],
                    [2, 5],
                    [4, 1],
                    [4, 2],
                    [5, 1],
                    [5, 2],
                    [6, 8],
                ],
                dtype=np.intp,
            )
            widget.labels_layer.data_setitem(
                (b_coordinates[:, 0], b_coordinates[:, 1]),
                1,
            )
            widget._select_label(1)
            self.assertTrue(
                np.all(
                    widget.labels_layer.data[
                        b_coordinates[:, 0],
                        b_coordinates[:, 1],
                    ]
                )
            )

            # Paint over an already-present hidden A membership.
            widget.labels_layer.data_setitem(
                (np.array([1]), np.array([1])),
                1,
            )
            self.assertEqual(widget.composite_layer.data[1, 1], 1)

            # Polygon over another hidden A region.
            widget.labels_layer.paint_polygon(
                np.array([[1, 4], [1, 5], [2, 5], [2, 4]]),
                1,
            )
            self.assertEqual(widget.composite_layer.data[1, 4], 1)

            # Fill follows the visible B component and raises all touched A
            # memberships without removing B underneath.
            widget.labels_layer.fill((4, 1), 1)
            np.testing.assert_array_equal(
                widget.composite_layer.data[4:6, 1:3],
                np.ones((2, 2), dtype=widget.composite_layer.data.dtype),
            )
            self.assertEqual(widget.composite_layer.data[6, 8], 2)
            for row, column in b_coordinates:
                self.assertIn(2, widget.overlap_store.memberships_at(row, column))

            # Native Undo/Redo includes top-only 1 -> 1 actions and restores
            # the exact visible class while retaining both memberships.
            widget.labels_layer.undo()
            np.testing.assert_array_equal(
                widget.composite_layer.data[4:6, 1:3],
                np.full((2, 2), 2, dtype=widget.composite_layer.data.dtype),
            )
            widget.labels_layer.redo()
            np.testing.assert_array_equal(
                widget.composite_layer.data[4:6, 1:3],
                np.ones((2, 2), dtype=widget.composite_layer.data.dtype),
            )

            # Erasing a hidden A leaves B visible; Undo must not accidentally
            # promote A when it restores the membership.
            widget.labels_layer.data_setitem(
                (np.array([6]), np.array([8])),
                0,
            )
            self.assertEqual(widget.composite_layer.data[6, 8], 2)
            widget.labels_layer.undo()
            self.assertIn(1, widget.overlap_store.memberships_at(6, 8))
            self.assertIn(2, widget.overlap_store.memberships_at(6, 8))
            self.assertEqual(widget.composite_layer.data[6, 8], 2)

            widget.save_labels()
            self.wait_for_save(widget)
            projection = iio.imread(labels_path)
            payload = read_embedded_annotations(labels_path)
            restored = OverlapStore.from_payload(payload, projection)
            self.assertEqual(restored.project()[4, 1], 1)
            self.assertEqual(restored.project()[6, 8], 2)
            self.assertEqual(restored.memberships_at(4, 1), (2, 1))
            self.assertEqual(restored.memberships_at(6, 8), (1, 2))
            saved_map, _ = read_class_config(mapping_path)
            self.assertIn(2, saved_map)

    def test_pick_uses_visible_semantic_class_without_binary_id_leak(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            widget._upsert_class(2, "B", "#123456")
            widget._upsert_class(300, "Review", "#abcdef")
            callback = widget.labels_layer._drag_modes[Mode.PICK]
            event = SimpleNamespace(
                position=(0, 0),
                view_direction=None,
                dims_displayed=(0, 1),
            )

            self.assertIsNot(
                widget.labels_layer._drag_modes,
                type(widget.labels_layer)._drag_modes,
            )
            with patch.object(
                widget.composite_layer,
                "get_value",
                return_value=2,
            ):
                callback(widget.labels_layer, event)
            self.assertEqual(widget.overlap_editor.active_class, 2)
            self.assertEqual(widget.labels_layer.selected_label, 1)

            with patch.object(
                widget.composite_layer,
                "get_value",
                return_value=300,
            ):
                callback(widget.labels_layer, event)
            self.assertEqual(widget.overlap_editor.active_class, 300)
            self.assertEqual(widget.labels_layer.selected_label, 1)

            with patch.object(
                widget.composite_layer,
                "get_value",
                return_value=0,
            ):
                callback(widget.labels_layer, event)
            self.assertEqual(widget.overlap_editor.active_class, 300)
            self.assertEqual(widget.labels_layer.selected_label, 0)

    def test_large_new_class_promotes_edit_and_save_dtypes_losslessly(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)

            self.assertEqual(widget.labels_layer.data.dtype, np.dtype(np.uint8))
            self.assertEqual(widget._labels_output_dtype, np.dtype(np.uint8))

            widget._upsert_class(300, "Review", "#abcdef")

            self.assertEqual(widget.labels_layer.data.dtype, np.dtype(np.uint8))
            self.assertEqual(widget.composite_layer.data.dtype, np.dtype(np.uint16))
            self.assertFalse(widget.composite_layer.editable)
            self.assertEqual(widget._labels_output_dtype, np.dtype(np.uint16))
            self.assertIs(widget.labels_layer.fill.__func__, overlap_fill)
            self.assertIs(
                widget.labels_layer.paint_polygon.__func__,
                fast_paint_polygon,
            )

            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                1,
            )
            self.assertEqual(widget.composite_layer.data[2, 3], 300)
            self.assertTrue(
                np.shares_memory(
                    widget.composite_layer.data,
                    widget.overlap_store.projection_view,
                )
            )
            np.testing.assert_allclose(
                widget.composite_layer.get_color(300),
                np.array([0xAB, 0xCD, 0xEF, 0xFF]) / 255,
                atol=1 / 255,
            )
            widget.save_labels()
            self.wait_for_save(widget)
            saved = iio.imread(labels_path)

            self.assertEqual(saved.dtype, np.dtype(np.uint16))
            self.assertEqual(saved[2, 3], 300)

    def test_png_class_300_promotes_saves_and_reloads_uint16(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(
                root,
                labels_suffix=".png",
            )

            widget._upsert_class(300, "Review", "#abcdef")
            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                1,
            )
            widget.save_labels()
            self.wait_for_save(widget)

            saved = iio.imread(labels_path)
            self.assertEqual(saved.dtype, np.dtype(np.uint16))
            self.assertEqual(saved[2, 3], 300)
            payload = read_embedded_annotations(labels_path)
            self.assertIsNotNone(payload)
            restored_store = OverlapStore.from_payload(payload, saved)
            self.assertIn(300, restored_store.memberships_at(2, 3))

            reloaded_viewer = ViewerModel()
            reloaded = LabelEditorWidget(reloaded_viewer)
            reloaded.image_line.setText(str(root / "image.png"))
            reloaded.label_line.setText(str(labels_path))
            reloaded.mapping_line.setText(str(mapping_path))
            reloaded.load_data()

            self.assertEqual(reloaded.composite_layer.data.dtype, np.uint16)
            self.assertEqual(reloaded.composite_layer.data[2, 3], 300)
            self.assertIn(300, reloaded.overlap_store.memberships_at(2, 3))

    def test_png_class_above_uint16_is_rejected_without_side_effects(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(
                root,
                labels_suffix=".png",
            )
            labels_before = labels_path.read_bytes()
            mapping_before = mapping_path.read_bytes()
            class_map_before = dict(widget.class_map)
            class_colors_before = dict(widget.class_colors)
            class_values_before = widget.overlap_store.class_values
            z_order_before = widget.overlap_store.z_order
            projection_before = widget.overlap_store.project()
            output_dtype_before = widget._labels_output_dtype
            active_before = widget.overlap_editor.active_class
            button_texts_before = [
                widget.class_button_layout.itemAt(index).widget().text()
                for index in range(widget.class_button_layout.count())
            ]

            with self.assertRaisesRegex(
                ValueError,
                "PNG supports class IDs up to 65535; use TIFF",
            ):
                widget._upsert_class(70000, "Too large", "#abcdef")

            self.assertEqual(labels_path.read_bytes(), labels_before)
            self.assertEqual(mapping_path.read_bytes(), mapping_before)
            self.assertEqual(widget.class_map, class_map_before)
            self.assertEqual(widget.class_colors, class_colors_before)
            self.assertEqual(widget.overlap_store.class_values, class_values_before)
            self.assertEqual(widget.overlap_store.z_order, z_order_before)
            np.testing.assert_array_equal(
                widget.overlap_store.project(),
                projection_before,
            )
            self.assertEqual(widget._labels_output_dtype, output_dtype_before)
            self.assertEqual(widget.overlap_editor.active_class, active_before)
            self.assertEqual(
                [
                    widget.class_button_layout.itemAt(index).widget().text()
                    for index in range(widget.class_button_layout.count())
                ],
                button_texts_before,
            )

    def test_delete_class_is_staged_until_save_then_updates_both_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)
            labels_before = labels_path.read_bytes()
            mapping_before = mapping_path.read_bytes()

            with patch.object(
                LabelEditorWidget,
                "_execute_dialog",
                return_value=True,
            ), patch(
                "napari_histo_label_editor._widget.atomic_save_labels"
            ) as save_labels, patch(
                "napari_histo_label_editor._widget.atomic_write_class_config"
            ) as write_class_config:
                widget._delete_class(1)

            save_labels.assert_not_called()
            write_class_config.assert_not_called()

            self.assertNotIn(1, widget.class_map)
            self.assertEqual(widget.labels_layer.data[1, 1], 0)
            self.assertTrue(widget._class_config_pending_save)
            self.assertEqual(widget.save_btn.text(), "Save class deletion [s]")
            self.assertEqual(labels_path.read_bytes(), labels_before)
            self.assertEqual(mapping_path.read_bytes(), mapping_before)

            widget.save_labels()
            self.wait_for_save(widget)

            self.assertEqual(iio.imread(labels_path)[1, 1], 0)
            saved_map, _ = read_class_config(mapping_path)
            self.assertNotIn(1, saved_map)
            self.assertFalse(widget._class_config_pending_save)
            self.assertEqual(widget.save_btn.text(), "Save [s]")

    def test_delete_class_can_reassign_pixels_to_an_existing_class(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)
            widget._upsert_class(2, "Stroma", "#123456")
            mapping_before = mapping_path.read_bytes()

            class FakeDeleteDialog:
                replacement_value = 2

                def __init__(self, *args, **kwargs):
                    del args, kwargs

            with patch(
                "napari_histo_label_editor._widget.DeleteClassDialog",
                FakeDeleteDialog,
            ), patch.object(
                LabelEditorWidget,
                "_execute_dialog",
                return_value=True,
            ):
                widget._delete_class(1)

            self.assertEqual(widget.labels_layer.data[1, 1], 1)
            self.assertEqual(widget.overlap_editor.active_class, 2)
            self.assertEqual(widget.labels_layer.selected_label, 1)
            self.assertEqual(mapping_path.read_bytes(), mapping_before)
            self.assertIn("Press Save", widget.viewer.status)

            widget.save_labels()
            self.wait_for_save(widget)

            self.assertEqual(iio.imread(labels_path)[1, 1], 2)
            saved_map, _ = read_class_config(mapping_path)
            self.assertEqual(saved_map[2], "Stroma")
            self.assertNotIn(1, saved_map)

    def test_delete_class_preserves_every_unrelated_pixel_exactly(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)
            widget._upsert_class(2, "Stroma", "#123456")
            original = np.array(
                [
                    [0, 1, 2, 0, 2, 1, 0, 2, 0, 1],
                    [2, 0, 0, 1, 0, 2, 2, 0, 1, 0],
                    [0, 2, 1, 2, 0, 0, 1, 0, 2, 2],
                    [1, 0, 2, 0, 1, 2, 0, 0, 2, 0],
                    [0, 0, 2, 1, 2, 0, 0, 1, 0, 2],
                    [2, 1, 0, 0, 2, 1, 2, 0, 0, 1],
                    [0, 2, 0, 1, 0, 2, 1, 2, 0, 0],
                    [1, 0, 2, 0, 0, 1, 0, 2, 2, 0],
                ],
                dtype=np.uint8,
            )
            widget.overlap_store.update_plane(1, original == 1)
            widget.overlap_store.update_plane(2, original == 2)
            widget.overlap_editor.refresh_projection()
            widget.overlap_editor.select_class(2)
            widget.labels_layer.refresh()
            widget.composite_layer.refresh()

            class FakeDeleteDialog:
                replacement_value = 2

                def __init__(self, *args, **kwargs):
                    del args, kwargs

            with patch(
                "napari_histo_label_editor._widget.DeleteClassDialog",
                FakeDeleteDialog,
            ), patch.object(
                LabelEditorWidget,
                "_execute_dialog",
                return_value=True,
            ):
                widget._delete_class(1)

            expected = original.copy()
            expected[original == 1] = 2
            np.testing.assert_array_equal(
                widget.overlap_store.project(),
                expected,
            )

            widget.save_labels()
            self.wait_for_save(widget)

            np.testing.assert_array_equal(iio.imread(labels_path), expected)
            saved_map, _ = read_class_config(mapping_path)
            self.assertEqual(saved_map, {0: "background", 2: "Stroma"})

    def test_deleting_unused_class_stays_staged_and_keeps_labels_identical(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)
            widget._upsert_class(2, "Unused", "#123456")
            labels_before = iio.imread(labels_path)
            label_bytes_before = labels_path.read_bytes()
            mapping_bytes_before = mapping_path.read_bytes()

            class FakeDeleteDialog:
                replacement_value = 0

                def __init__(self, *args, **kwargs):
                    del args, kwargs

            with patch(
                "napari_histo_label_editor._widget.DeleteClassDialog",
                FakeDeleteDialog,
            ), patch.object(
                LabelEditorWidget,
                "_execute_dialog",
                return_value=True,
            ):
                widget._delete_class(2)

            np.testing.assert_array_equal(
                widget.overlap_store.project(),
                labels_before,
            )
            self.assertEqual(labels_path.read_bytes(), label_bytes_before)
            self.assertEqual(mapping_path.read_bytes(), mapping_bytes_before)
            self.assertTrue(widget._class_config_pending_save)
            self.assertIn("not used by any pixels", widget.viewer.status)

            widget.save_labels()
            self.wait_for_save(widget)

            np.testing.assert_array_equal(iio.imread(labels_path), labels_before)
            saved_map, _ = read_class_config(mapping_path)
            self.assertNotIn(2, saved_map)

    def test_label_save_failure_leaves_pending_class_csv_untouched(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)
            label_bytes_before = labels_path.read_bytes()
            mapping_bytes_before = mapping_path.read_bytes()
            with patch.object(
                LabelEditorWidget,
                "_execute_dialog",
                return_value=True,
            ):
                widget._delete_class(1)

            worker = FakeSaveWorker()
            with patch(
                "napari_histo_label_editor._widget.create_worker",
                return_value=worker,
            ), patch(
                "napari_histo_label_editor._widget.atomic_write_class_config"
            ) as write_class_config, patch(
                "napari_histo_label_editor._widget.QMessageBox.critical"
            ):
                widget.save_labels()
                worker.errored.emit(RuntimeError("simulated label failure"))
                worker.finished.emit()

            write_class_config.assert_not_called()
            self.assertEqual(labels_path.read_bytes(), label_bytes_before)
            self.assertEqual(mapping_path.read_bytes(), mapping_bytes_before)
            self.assertTrue(widget._class_config_pending_save)
            self.assertEqual(widget.save_btn.text(), "Save class deletion [s]")
            self.assertIn("simulated label failure", widget.viewer.status)

    def test_external_label_replacement_blocks_pending_delete_save(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)
            mapping_bytes_before = mapping_path.read_bytes()
            with patch.object(
                LabelEditorWidget,
                "_execute_dialog",
                return_value=True,
            ):
                widget._delete_class(1)

            external_labels = np.full((8, 10), 7, dtype=np.uint8)
            replacement_path = root / "external-labels.tif"
            iio.imwrite(replacement_path, external_labels)
            os.replace(replacement_path, labels_path)

            with patch(
                "napari_histo_label_editor._widget.QFileDialog.getSaveFileName",
                return_value=("", ""),
            ) as save_as_dialog, patch(
                "napari_histo_label_editor._widget.create_worker"
            ) as create_worker, patch(
                "napari_histo_label_editor._widget.atomic_write_class_config"
            ) as write_class_config:
                widget.save_labels()

            save_as_dialog.assert_called_once()
            create_worker.assert_not_called()
            write_class_config.assert_not_called()
            np.testing.assert_array_equal(iio.imread(labels_path), external_labels)
            self.assertEqual(mapping_path.read_bytes(), mapping_bytes_before)
            self.assertTrue(widget._class_config_pending_save)
            self.assertTrue(widget.save_btn.isEnabled())
            self.assertEqual(widget.save_btn.text(), "Save As… [s]")

    def test_save_as_preserves_pending_deletion_without_touching_changed_csv(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)
            label_bytes_before = labels_path.read_bytes()
            with patch.object(
                LabelEditorWidget,
                "_execute_dialog",
                return_value=True,
            ):
                widget._delete_class(1)

            external_mapping_path = root / "external-mapping.csv"
            pd.DataFrame(
                {
                    "value": [0, 99],
                    "name": ["background", "External"],
                }
            ).to_csv(external_mapping_path, index=False)
            external_mapping_bytes = external_mapping_path.read_bytes()
            os.replace(external_mapping_path, mapping_path)
            rescued_path = root / "rescued-labels.tif"

            with patch(
                "napari_histo_label_editor._widget.QFileDialog.getSaveFileName",
                return_value=(str(rescued_path), ""),
            ), patch(
                "napari_histo_label_editor._widget.atomic_write_class_config"
            ) as write_class_config:
                widget.save_labels()
                self.wait_for_save(widget)

            write_class_config.assert_not_called()
            self.assertEqual(labels_path.read_bytes(), label_bytes_before)
            self.assertEqual(mapping_path.read_bytes(), external_mapping_bytes)
            self.assertTrue(rescued_path.is_file())
            self.assertEqual(iio.imread(rescued_path)[1, 1], 0)
            self.assertIsNotNone(read_embedded_annotations(rescued_path))
            self.assertEqual(widget.labels_path, rescued_path.resolve())
            self.assertEqual(widget.label_line.text(), str(rescued_path.resolve()))
            self.assertTrue(widget._class_config_pending_save)
            self.assertIn("mapping CSV was not changed", widget.viewer.status)
            self.assertEqual(widget.save_btn.text(), "Save As… [s]")

    def test_external_mapping_change_during_confirmation_is_exact_no_op(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)
            labels_before = np.array(widget.labels_layer.data, copy=True)
            class_map_before = dict(widget.class_map)
            class_colors_before = dict(widget.class_colors)
            label_bytes_before = labels_path.read_bytes()

            external_mapping_path = root / "external-mapping.csv"
            pd.DataFrame(
                {
                    "value": [0, 8],
                    "name": ["background", "External"],
                }
            ).to_csv(external_mapping_path, index=False)
            external_mapping_bytes = external_mapping_path.read_bytes()

            def replace_mapping_and_accept(*args):
                del args
                os.replace(external_mapping_path, mapping_path)
                return True

            with patch.object(
                LabelEditorWidget,
                "_execute_dialog",
                side_effect=replace_mapping_and_accept,
            ), patch(
                "napari_histo_label_editor._widget.QMessageBox.critical"
            ):
                widget._delete_class(1)

            np.testing.assert_array_equal(widget.labels_layer.data, labels_before)
            self.assertEqual(widget.class_map, class_map_before)
            self.assertEqual(widget.class_colors, class_colors_before)
            self.assertEqual(labels_path.read_bytes(), label_bytes_before)
            self.assertEqual(mapping_path.read_bytes(), external_mapping_bytes)
            self.assertFalse(widget._class_config_pending_save)
            self.assertIn("missing or changed", widget.viewer.status)

    def test_delete_button_tracks_selected_mapped_nonbackground_class(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))

            self.assertEqual(widget.labels_layer.selected_label, 1)
            self.assertTrue(widget.delete_class_btn.isEnabled())

            widget.labels_layer.selected_label = 0
            self.assertTrue(widget.delete_class_btn.isEnabled())

            widget.labels_layer.selected_label = 7
            self.assertEqual(widget.labels_layer.selected_label, 0)
            self.assertTrue(widget.delete_class_btn.isEnabled())

            widget.labels_layer.selected_label = 1
            self.assertTrue(widget.delete_class_btn.isEnabled())

    def test_load_selects_a_mapped_value_when_class_one_is_absent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            labels_path = root / "labels.tif"
            mapping_path = root / "mapping.csv"
            iio.imwrite(image_path, np.zeros((8, 10, 3), dtype=np.uint8))
            labels = np.zeros((8, 10), dtype=np.uint8)
            labels[1, 1] = 23
            iio.imwrite(labels_path, labels)
            pd.DataFrame(
                {
                    "value": [0, 23],
                    "name": ["background", "Tumor"],
                }
            ).to_csv(mapping_path, index=False)

            viewer = ViewerModel()
            widget = LabelEditorWidget(viewer)
            widget.image_line.setText(str(image_path))
            widget.label_line.setText(str(labels_path))
            widget.mapping_line.setText(str(mapping_path))
            widget.load_data()

            self.assertEqual(widget.overlap_editor.active_class, 23)
            self.assertEqual(widget.labels_layer.selected_label, 1)
            self.assertTrue(widget.delete_class_btn.isEnabled())

    def test_deleted_value_reintroduced_before_save_is_sanitized(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)
            with patch.object(
                LabelEditorWidget,
                "_execute_dialog",
                return_value=True,
            ):
                widget._delete_class(1)

            # Defense in depth: even direct/plugin code cannot smuggle the
            # removed value back into the saved mask after deletion.
            widget.labels_layer.data_setitem(
                (np.array([3]), np.array([4])),
                1,
            )
            self.assertEqual(widget.labels_layer.data[3, 4], 1)

            widget.save_labels()
            self.wait_for_save(widget)

            saved = iio.imread(labels_path)
            self.assertFalse(np.any(saved == 1))
            saved_map, _ = read_class_config(mapping_path)
            self.assertNotIn(1, saved_map)

    def test_loading_again_discards_unsaved_class_deletion_state(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)
            label_bytes_before = labels_path.read_bytes()
            mapping_bytes_before = mapping_path.read_bytes()
            with patch.object(
                LabelEditorWidget,
                "_execute_dialog",
                return_value=True,
            ):
                widget._delete_class(1)

            self.assertTrue(widget._class_config_pending_save)
            self.assertNotIn(1, widget.class_map)
            self.assertEqual(widget.labels_layer.data[1, 1], 0)

            widget.load_data()

            self.assertFalse(widget._class_config_pending_save)
            self.assertIn(1, widget.class_map)
            self.assertEqual(widget.labels_layer.data[1, 1], 1)
            self.assertEqual(widget.labels_layer.selected_label, 1)
            self.assertTrue(widget.delete_class_btn.isEnabled())
            self.assertEqual(widget.save_btn.text(), "Save [s]")
            self.assertEqual(labels_path.read_bytes(), label_bytes_before)
            self.assertEqual(mapping_path.read_bytes(), mapping_bytes_before)

    def test_cancel_delete_is_an_exact_no_op(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)
            labels_before = np.array(widget.labels_layer.data, copy=True)
            class_map_before = dict(widget.class_map)
            class_colors_before = dict(widget.class_colors)
            label_bytes_before = labels_path.read_bytes()
            mapping_bytes_before = mapping_path.read_bytes()

            with patch.object(
                LabelEditorWidget,
                "_execute_dialog",
                return_value=False,
            ):
                widget._delete_class(1)

            np.testing.assert_array_equal(widget.labels_layer.data, labels_before)
            self.assertEqual(widget.class_map, class_map_before)
            self.assertEqual(widget.class_colors, class_colors_before)
            self.assertFalse(widget._class_config_pending_save)
            self.assertEqual(labels_path.read_bytes(), label_bytes_before)
            self.assertEqual(mapping_path.read_bytes(), mapping_bytes_before)

    def test_background_cannot_be_deleted(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, mapping_path = self.load_small_project(Path(tmp))
            mapping_before = mapping_path.read_bytes()

            with patch(
                "napari_histo_label_editor._widget.DeleteClassDialog"
            ) as delete_dialog:
                widget._delete_class(0)

            delete_dialog.assert_not_called()
            self.assertIn(0, widget.class_map)
            self.assertFalse(widget._class_config_pending_save)
            self.assertEqual(mapping_path.read_bytes(), mapping_before)
            self.assertIn("cannot be deleted", widget.viewer.status)

    def test_failed_class_csv_phase_leaves_safe_extra_mapping_and_can_retry(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, mapping_path = self.load_small_project(root)
            with patch.object(
                LabelEditorWidget,
                "_execute_dialog",
                return_value=True,
            ):
                widget._delete_class(1)

            with patch(
                "napari_histo_label_editor._widget.atomic_write_class_config",
                side_effect=RuntimeError("simulated CSV failure"),
            ), patch(
                "napari_histo_label_editor._widget.QMessageBox.critical"
            ):
                widget.save_labels()
                self.wait_for_save(widget)

            self.assertEqual(iio.imread(labels_path)[1, 1], 0)
            saved_map, _ = read_class_config(mapping_path)
            self.assertIn(1, saved_map)
            self.assertTrue(widget._class_config_pending_save)
            self.assertIn("class CSV was not updated", widget.viewer.status)

            widget.save_labels()
            self.wait_for_save(widget)
            saved_map, _ = read_class_config(mapping_path)
            self.assertNotIn(1, saved_map)
            self.assertFalse(widget._class_config_pending_save)

    def test_class_scan_and_reassignment_use_bounded_chunks(self):
        labels = np.arange(60, dtype=np.uint16).reshape(6, 10) % 3
        original = labels.copy()
        expected_count = int(np.count_nonzero(labels == 1))

        count = LabelEditorWidget._count_label_pixels(
            labels[:, ::-1],
            1,
            mask_bytes=20,
        )
        changed = LabelEditorWidget._reassign_label_pixels(
            labels,
            1,
            2,
            mask_bytes=20,
        )

        self.assertEqual(count, expected_count)
        self.assertEqual(changed, expected_count)
        expected = original.copy()
        expected[original == 1] = 2
        np.testing.assert_array_equal(labels, expected)

    def test_pending_delete_blocks_more_class_definition_changes(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            with patch.object(
                LabelEditorWidget,
                "_execute_dialog",
                return_value=True,
            ):
                widget._delete_class(1)

            self.assertFalse(widget.add_class_btn.isEnabled())
            self.assertFalse(widget.edit_class_btn.isEnabled())
            self.assertFalse(widget.delete_class_btn.isEnabled())
            self.assertTrue(widget.class_button_container.isEnabled())
            with self.assertRaisesRegex(ValueError, "pending class deletion"):
                widget._upsert_class(2, "Stroma", "#123456")


if __name__ == "__main__":
    unittest.main()
