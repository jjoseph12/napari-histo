import os
import unittest
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import patch

import imageio.v3 as iio
import numpy as np
import pandas as pd
from napari.components import ViewerModel
from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari.layers import Labels
from napari.layers.labels._labels_constants import Mode
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication, QDockWidget, QMessageBox, QWidget

from napari_histo_label_editor._embedded_annotations import (
    read_embedded_annotations,
)
from napari_histo_label_editor._fast_fill import overlap_fill
from napari_histo_label_editor._fast_polygon import fast_paint_polygon
from napari_histo_label_editor._class_config import read_class_config
from napari_histo_label_editor._overlap_store import OverlapStore
from napari_histo_label_editor._widget import (
    LabelEditorWidget,
    SELECTION_ACTIONS_DOCK_NAME,
)


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

    def test_selection_actions_use_one_compact_companion_dock_lifecycle(self):
        class FakeWindow:
            def __init__(self):
                self.dock_widgets = {
                    SELECTION_ACTIONS_DOCK_NAME: QWidget()
                }
                self.docks = {}
                self.add_calls = []
                self.remove_calls = []

            def add_dock_widget(self, widget, **kwargs):
                self.add_calls.append((widget, kwargs))
                name = kwargs["name"]
                dock = QDockWidget(name)
                dock.setWidget(widget)
                self.dock_widgets[name] = widget
                self.docks[name] = dock
                return dock

            def remove_dock_widget(self, widget):
                self.remove_calls.append(widget)
                for name, candidate in tuple(self.dock_widgets.items()):
                    if candidate is not widget:
                        continue
                    self.dock_widgets.pop(name)
                    dock = self.docks.pop(name, None)
                    if dock is not None:
                        widget.setParent(None)
                        dock.deleteLater()
                    return
                raise LookupError(widget)

        original_viewer = ViewerModel()
        widget = LabelEditorWidget(original_viewer)
        self.assertEqual(widget.layout().indexOf(widget.selection_actions_widget), -1)
        self.assertEqual(
            [
                widget.selection_actions_widget.layout().itemAt(index).widget()
                for index in range(
                    widget.selection_actions_widget.layout().count()
                )
            ],
            [widget.delete_region_btn, widget.clear_region_btn],
        )
        self.assertIsNone(widget._selection_actions_dock)

        owner = QDockWidget("editor")
        owner.setWidget(widget)
        fake_window = FakeWindow()
        stale = fake_window.dock_widgets[SELECTION_ACTIONS_DOCK_NAME]
        widget.viewer = SimpleNamespace(window=fake_window)

        widget._install_selection_actions_dock()
        self.assertEqual(fake_window.remove_calls, [stale])
        self.assertEqual(len(fake_window.add_calls), 1)
        _, kwargs = fake_window.add_calls[0]
        self.assertEqual(kwargs["area"], "left")
        self.assertEqual(kwargs["allowed_areas"], ["left"])
        self.assertFalse(kwargs["add_vertical_stretch"])
        self.assertEqual(
            widget._selection_actions_dock.minimumHeight(),
            widget._selection_actions_dock.maximumHeight(),
        )
        self.assertLessEqual(widget._selection_actions_dock.maximumHeight(), 90)
        self.assertIs(
            widget._selection_actions_dock.titleBarWidget(),
            widget.selection_actions_widget,
        )
        self.assertIs(
            widget._selection_actions_dock.widget(),
            widget._selection_actions_body,
        )
        self.assertEqual(widget._selection_actions_body.height(), 0)

        # napari replaces custom title chrome whenever a dock is shown. The
        # deferred restoration must put the same live buttons back without
        # changing their action state.
        widget.delete_region_btn.setEnabled(True)
        widget._selection_actions_dock.setTitleBarWidget(
            QWidget(widget._selection_actions_dock)
        )
        widget._schedule_selection_actions_title_bar(True)
        self.app.processEvents()
        self.assertIs(
            widget._selection_actions_dock.titleBarWidget(),
            widget.selection_actions_widget,
        )
        self.assertTrue(widget.delete_region_btn.isEnabled())

        first_dock = widget._selection_actions_dock
        widget._install_selection_actions_dock()
        self.assertIs(widget._selection_actions_dock, first_dock)
        self.assertEqual(len(fake_window.add_calls), 1)

        # napari detaches the editor before deleting its wrapping dock.  The
        # companion must leave the public dock mapping at the same time.
        widget.setParent(None)
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.assertNotIn(
            SELECTION_ACTIONS_DOCK_NAME,
            fake_window.dock_widgets,
        )
        self.assertIs(widget.selection_actions_widget.parentWidget(), widget)
        self.assertIs(widget._selection_actions_body.parentWidget(), widget)
        self.assertEqual(widget.delete_region_btn.text(), "Delete…")
        self.assertEqual(widget.clear_region_btn.text(), "Clear")

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

    @staticmethod
    def pick_connected_region(widget, position):
        callback = widget.labels_layer._drag_modes[Mode.PICK]
        callback(
            widget.labels_layer,
            SimpleNamespace(
                position=position,
                view_direction=None,
                dims_displayed=(0, 1),
            ),
        )
        return widget._annotation_selection

    @staticmethod
    def connected_runtime_state(widget):
        """Return an exact small-project snapshot for rejected UI actions."""

        tracker = widget.labels_layer._napari_histo_edit_tracker
        selection = widget._annotation_selection
        outlines = ()
        if widget._selection_layer_is_present():
            outlines = tuple(
                np.asarray(path).tobytes()
                for path in widget._selection_layer.data
            )
        return (
            widget.overlap_store._packed_masks.tobytes(),
            widget.overlap_editor.composite.tobytes(),
            np.asarray(widget.labels_layer.data).tobytes(),
            widget.overlap_store.revision,
            widget.overlap_store.generation,
            widget.overlap_store.projection_sha256,
            tuple(id(item) for item in widget.labels_layer._undo_history),
            tuple(id(item) for item in widget.labels_layer._redo_history),
            tuple(id(item) for item in tracker.undo_items),
            tuple(id(item) for item in tracker.redo_items),
            id(selection) if selection is not None else None,
            outlines,
        )

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
                    "Histology",
                    "Annotations",
                    "Annotation tools — 1: tumor",
                    "Selected annotation outline (preview)",
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

    def test_editable_destination_saves_directly_without_changing_load_input(
        self,
    ):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)
            source_bytes = labels_path.read_bytes()
            source_input = widget.label_line.text()
            typed_destination = root / "typed-copy.tif"
            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                1,
            )

            self.assertFalse(widget.save_destination_line.isReadOnly())
            self.assertTrue(widget.save_destination_line.isEnabled())
            self.assertEqual(
                widget.save_destination_line.text(),
                str(labels_path.resolve()),
            )

            widget.save_destination_line.setText(str(typed_destination))

            self.assertEqual(widget.save_btn.text(), "Save As [s]")
            self.assertEqual(widget.label_line.text(), source_input)
            self.assertTrue(widget.class_button_container.isEnabled())
            with patch(
                "napari_histo_label_editor._widget.QFileDialog.getSaveFileName"
            ) as save_as_dialog:
                widget.save_labels()
                self.wait_for_save(widget)

            save_as_dialog.assert_not_called()
            self.assertEqual(labels_path.read_bytes(), source_bytes)
            self.assertEqual(widget.labels_path, typed_destination.resolve())
            self.assertEqual(widget._loaded_labels_path, labels_path.resolve())
            self.assertEqual(widget.label_line.text(), source_input)
            self.assertEqual(
                widget.save_destination_line.text(),
                str(typed_destination.resolve()),
            )
            self.assertEqual(widget.save_btn.text(), "Save [s]")
            projection = iio.imread(typed_destination)
            self.assertEqual(projection[2, 3], 1)
            self.assertIsNotNone(read_embedded_annotations(typed_destination))

            widget.labels_layer.data_setitem(
                (np.array([3]), np.array([4])),
                1,
            )
            with patch(
                "napari_histo_label_editor._widget.QFileDialog.getSaveFileName"
            ) as save_as_dialog:
                widget.save_labels()
                self.wait_for_save(widget)
            save_as_dialog.assert_not_called()
            self.assertEqual(iio.imread(typed_destination)[3, 4], 1)

    def test_typed_existing_destination_requires_confirmation_and_cancel_is_safe(
        self,
    ):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)
            existing = root / "existing.tif"
            external = np.full((8, 10), 47, dtype=np.uint8)
            iio.imwrite(existing, external)
            external_bytes = existing.read_bytes()
            adopted_path = widget.labels_path
            adopted_identity = widget._labels_destination_identity
            adopted_dtype = widget._labels_output_dtype
            source_input = widget.label_line.text()
            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                1,
            )
            widget.save_destination_line.setText(str(existing))

            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
            with patch(
                "napari_histo_label_editor._widget.QMessageBox.question",
                return_value=getattr(buttons, "No"),
            ) as confirmation, patch(
                "napari_histo_label_editor._widget.create_worker"
            ) as create_worker:
                widget.save_labels()

            confirmation.assert_called_once()
            create_worker.assert_not_called()
            self.assertEqual(existing.read_bytes(), external_bytes)
            np.testing.assert_array_equal(iio.imread(existing), external)
            self.assertEqual(widget.labels_path, adopted_path)
            self.assertEqual(
                widget._labels_destination_identity,
                adopted_identity,
            )
            self.assertEqual(widget._labels_output_dtype, adopted_dtype)
            self.assertEqual(widget.label_line.text(), source_input)
            self.assertEqual(
                widget.save_destination_line.text(),
                str(existing),
            )
            self.assertEqual(widget.labels_layer.data[2, 3], 1)
            self.assertEqual(labels_path.resolve(), adopted_path)

    def test_confirmed_typed_existing_destination_is_verified_and_adopted(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)
            existing = root / "existing.tif"
            iio.imwrite(existing, np.full((8, 10), 52, dtype=np.uint8))
            source_bytes = labels_path.read_bytes()
            source_input = widget.label_line.text()
            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                1,
            )
            widget.save_destination_line.setText(str(existing))
            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)

            with patch(
                "napari_histo_label_editor._widget.QMessageBox.question",
                return_value=getattr(buttons, "Yes"),
            ) as confirmation:
                widget.save_labels()
                self.wait_for_save(widget)

            confirmation.assert_called_once()
            self.assertEqual(labels_path.read_bytes(), source_bytes)
            self.assertEqual(widget.labels_path, existing.resolve())
            self.assertEqual(widget.label_line.text(), source_input)
            self.assertEqual(widget._loaded_labels_path, labels_path.resolve())
            self.assertEqual(
                widget.save_destination_line.text(),
                str(existing.resolve()),
            )
            projection = iio.imread(existing)
            self.assertEqual(projection[2, 3], 1)
            self.assertIsNotNone(read_embedded_annotations(existing))

    def test_typed_existing_destination_race_preserves_external_winner(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)
            existing = root / "existing.tif"
            iio.imwrite(existing, np.full((8, 10), 7, dtype=np.uint8))
            external_winner = np.full((8, 10), 83, dtype=np.uint8)
            adopted_path = widget.labels_path
            adopted_identity = widget._labels_destination_identity
            source_bytes = labels_path.read_bytes()
            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                1,
            )
            widget.save_destination_line.setText(str(existing))
            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)

            def replace_during_confirmation(*args):
                del args
                replacement = root / "external-winner.tif"
                iio.imwrite(replacement, external_winner)
                os.replace(replacement, existing)
                return getattr(buttons, "Yes")

            with patch(
                "napari_histo_label_editor._widget.QMessageBox.question",
                side_effect=replace_during_confirmation,
            ), patch(
                "napari_histo_label_editor._widget.QMessageBox.critical"
            ):
                widget.save_labels()
                self.wait_for_save(widget)

            np.testing.assert_array_equal(iio.imread(existing), external_winner)
            self.assertEqual(labels_path.read_bytes(), source_bytes)
            self.assertEqual(widget.labels_path, adopted_path)
            self.assertEqual(
                widget._labels_destination_identity,
                adopted_identity,
            )
            self.assertEqual(
                widget.save_destination_line.text(),
                str(existing.resolve()),
            )
            self.assertEqual(widget.labels_layer.data[2, 3], 1)
            self.assertIn("changed since", widget.viewer.status)

    def test_typed_new_destination_race_never_overwrites_intruder(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)
            destination = root / "new-target.tif"
            external_winner = np.full((8, 10), 91, dtype=np.uint8)
            adopted_path = widget.labels_path
            adopted_identity = widget._labels_destination_identity
            source_bytes = labels_path.read_bytes()
            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                1,
            )
            widget.save_destination_line.setText(str(destination))
            real_link = os.link

            def create_intruder_then_link(source, target):
                iio.imwrite(target, external_winner)
                return real_link(source, target)

            with patch(
                "napari_histo_label_editor._io.os.link",
                side_effect=create_intruder_then_link,
            ), patch(
                "napari_histo_label_editor._widget.QMessageBox.critical"
            ):
                widget.save_labels()
                self.wait_for_save(widget)

            np.testing.assert_array_equal(
                iio.imread(destination),
                external_winner,
            )
            self.assertEqual(labels_path.read_bytes(), source_bytes)
            self.assertEqual(widget.labels_path, adopted_path)
            self.assertEqual(
                widget._labels_destination_identity,
                adopted_identity,
            )
            self.assertEqual(
                widget.save_destination_line.text(),
                str(destination.resolve()),
            )
            self.assertEqual(widget.labels_layer.data[2, 3], 1)
            self.assertIn("appeared while", widget.viewer.status)

    def test_typed_png_destination_adopts_lossless_promoted_dtype(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(
                root,
                labels_dtype=np.int32,
            )
            source_input = widget.label_line.text()
            widget._upsert_class(300, "Large class", "#123456")
            widget._select_label(300)
            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                1,
            )
            destination = root / "typed-copy.png"
            widget.save_destination_line.setText(str(destination))

            widget.save_labels()
            self.wait_for_save(widget)

            projection = iio.imread(destination)
            self.assertEqual(projection.dtype, np.dtype(np.uint16))
            self.assertEqual(projection[2, 3], 300)
            self.assertEqual(widget._labels_output_dtype, np.dtype(np.uint16))
            self.assertEqual(widget.labels_path, destination.resolve())
            self.assertEqual(widget._loaded_labels_path, labels_path.resolve())
            self.assertEqual(widget.label_line.text(), source_input)
            payload = read_embedded_annotations(destination)
            self.assertIsNotNone(payload)
            restored = OverlapStore.from_payload(payload, projection=projection)
            self.assertEqual(restored.memberships_at(2, 3), (300,))

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
            self.assertEqual(widget.label_line.text(), str(labels_path.resolve()))
            self.assertEqual(widget._loaded_labels_path, labels_path.resolve())
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
            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                0,
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

    def test_native_labels_controls_bridge_opacity_and_large_brush_range(self):
        viewer = ViewerModel()
        empty_widget = LabelEditorWidget(viewer)
        self.assertFalse(hasattr(empty_widget, "overlay_opacity_slider"))
        layout = empty_widget.layout()
        load_index = layout.indexOf(empty_widget.load_btn)
        self.assertEqual(
            layout.itemAt(load_index + 1).widget().text(),
            "Classes",
        )

        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            self.assertAlmostEqual(widget.composite_layer.opacity, 0.45)
            self.assertEqual(widget.labels_layer.opacity, 0.0)
            controls = QtLabelsControls(widget.labels_layer)
            try:
                self.assertTrue(widget._install_native_labels_controls(controls))
                opacity_slider = (
                    controls._opacity_blending_controls.opacity_slider
                )
                brush_slider = (
                    controls._brush_size_slider_control.brush_size_slider
                )
                semantic_spinbox = (
                    controls._label_control.selection_spinbox
                )
                self.assertAlmostEqual(opacity_slider.value(), 0.45)
                self.assertEqual(brush_slider.minimum(), 1)
                self.assertEqual(brush_slider.maximum(), 512)
                self.assertIn("1–512", brush_slider.toolTip())
                self.assertEqual(semantic_spinbox.value(), 1)
                self.assertEqual(widget.labels_layer.selected_label, 1)
                self.assertIn(
                    "annotation class",
                    semantic_spinbox.toolTip(),
                )

                widget._upsert_class(23, "Tumor edge", "#123456", persist=False)
                widget._upsert_class(300, "Review", "#abcdef", persist=False)
                self.assertEqual(semantic_spinbox.minimum(), 0)
                self.assertEqual(semantic_spinbox.maximum(), 300)
                self.assertEqual(semantic_spinbox.value(), 300)
                self.assertEqual(widget.labels_layer.selected_label, 1)

                # Semantic selection never writes a high ID into the binary
                # proxy and does not trigger a complete composite refresh.
                with patch.object(
                    widget.composite_layer,
                    "refresh",
                    side_effect=AssertionError(
                        "semantic class selection refreshed the full slide"
                    ),
                ):
                    semantic_spinbox.setValue(23)
                self.assertEqual(widget.overlap_editor.active_class, 23)
                self.assertEqual(widget.labels_layer.selected_label, 1)
                self.assertEqual(semantic_spinbox.value(), 23)
                np.testing.assert_allclose(
                    widget.labels_layer._selected_color,
                    widget._label_rgba(23),
                )

                # Native +/- navigation skips gaps between mapped IDs.
                semantic_spinbox.stepBy(1)
                self.assertEqual(semantic_spinbox.value(), 300)
                self.assertEqual(widget.overlap_editor.active_class, 300)
                self.assertEqual(widget.labels_layer.selected_label, 1)
                semantic_spinbox.stepBy(-1)
                self.assertEqual(semantic_spinbox.value(), 23)
                self.assertEqual(widget.overlap_editor.active_class, 23)

                # Unknown typed IDs restore the real class without changing
                # the proxy, store, or active semantic class.
                revision_before = widget.overlap_store.revision
                semantic_spinbox.setValue(17)
                self.assertEqual(semantic_spinbox.value(), 23)
                self.assertEqual(widget.overlap_editor.active_class, 23)
                self.assertEqual(widget.labels_layer.selected_label, 1)
                self.assertEqual(widget.overlap_store.revision, revision_before)
                self.assertIn("Class 17 is not available", widget.viewer.status)

                semantic_spinbox.setValue(0)
                self.assertEqual(widget.labels_layer.selected_label, 0)
                self.assertEqual(widget.overlap_editor.active_class, 23)
                self.assertEqual(semantic_spinbox.value(), 0)
                semantic_spinbox.stepBy(1)
                self.assertEqual(widget.overlap_editor.active_class, 1)
                self.assertEqual(widget.labels_layer.selected_label, 1)
                self.assertEqual(semantic_spinbox.value(), 1)

                # Binary data events must not reset the decoupled selector's
                # range back to uint8, and installing twice is idempotent.
                widget.labels_layer.events.data()
                self.assertEqual(semantic_spinbox.maximum(), 300)
                self.assertTrue(widget._install_native_labels_controls(controls))
                with patch.object(
                    widget,
                    "_select_label",
                    wraps=widget._select_label,
                ) as select_label:
                    semantic_spinbox.setValue(23)
                select_label.assert_called_once_with(23)
                self.assertEqual(widget.labels_layer.selected_label, 1)

                proxy_opacity_events = []
                widget.labels_layer.events.opacity.connect(
                    lambda event: proxy_opacity_events.append(
                        widget.labels_layer.opacity
                    )
                )
                with patch.object(
                    widget.composite_layer,
                    "refresh",
                    side_effect=AssertionError(
                        "opacity change refreshed the full annotation layer"
                    ),
                ):
                    opacity_slider.setValue(0.67)
                self.assertAlmostEqual(widget.composite_layer.opacity, 0.67)
                self.assertAlmostEqual(opacity_slider.value(), 0.67)
                self.assertEqual(widget.labels_layer.opacity, 0.0)
                self.assertEqual(proxy_opacity_events, [])

                brush_slider.setValue(401)
                self.assertEqual(widget.labels_layer.brush_size, 401)

                # Programmatic proxy opacity cannot expose the binary edit
                # mask or change visible annotation opacity.
                widget.labels_layer.opacity = 0.8
                self.assertAlmostEqual(widget.composite_layer.opacity, 0.67)
                self.assertAlmostEqual(opacity_slider.value(), 0.67)
                self.assertEqual(widget.labels_layer.opacity, 0.0)

                # Selecting the visible composite and using its own native
                # slider also updates the cached edit-layer control.
                widget.composite_layer.opacity = 0.32
                self.assertAlmostEqual(opacity_slider.value(), 0.32)
                self.assertAlmostEqual(widget._annotation_opacity, 0.32)

                # The deferred real-window retry is harmless and does not
                # duplicate the slider connection.
                self.assertTrue(widget._install_native_labels_controls(controls))
                opacity_slider.setValue(0.28)
                self.assertAlmostEqual(widget.composite_layer.opacity, 0.28)
                self.assertEqual(widget.labels_layer.opacity, 0.0)
            finally:
                controls.close()

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

            # The semantic eraser removes visible B even while A is selected,
            # revealing A without deleting its hidden membership.
            widget.labels_layer.data_setitem(
                (np.array([6]), np.array([8])),
                0,
            )
            self.assertEqual(widget.composite_layer.data[6, 8], 1)
            self.assertEqual(
                widget.overlap_store.memberships_at(6, 8),
                (1,),
            )
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

    def test_semantic_eraser_mixed_stroke_is_sparse_and_undo_redo_exact(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            base = (np.array([1, 1, 1, 1]), np.array([1, 2, 3, 4]))
            widget.labels_layer.data_setitem(base, 1)
            widget._upsert_class(2, "B", "#123456")
            widget.labels_layer.data_setitem(
                (np.array([1, 1]), np.array([1, 2])),
                1,
            )
            widget._upsert_class(3, "C", "#abcdef")
            widget.labels_layer.data_setitem(
                (np.array([1, 1]), np.array([2, 3])),
                1,
            )
            # Class 25 is intentionally unrelated to every touched semantic
            # pixel, reproducing the user's active-class mismatch.
            widget._upsert_class(25, "Unrelated", "#654321")
            self.assertEqual(widget.overlap_editor.active_class, 25)
            self.assertFalse(np.any(widget.labels_layer.data))
            widget._select_label(0)
            self.assertIn("visible/top", widget.viewer.status)
            self.assertEqual(
                widget.labels_layer.features.iloc[0]["Label"],
                "Erase visible/top annotation",
            )

            store = widget.overlap_store
            packed_before = np.array(store.packed_masks, copy=True)
            projection_before = np.array(store.projection_view, copy=True)
            edit_before = np.array(widget.labels_layer.data, copy=True)
            rows = np.array([1, 1, 1, 1, 0, 1])
            columns = np.array([1, 2, 3, 4, 0, 1])

            with patch.object(
                store,
                "select_plane",
                side_effect=AssertionError("eraser unpacked a full plane"),
            ), patch.object(
                store,
                "project",
                side_effect=AssertionError("eraser projected the full slide"),
            ), patch.object(
                widget.overlap_editor,
                "full_sync",
                side_effect=AssertionError("eraser performed a full sync"),
            ), patch.object(
                widget,
                "_refresh_composite_bounds",
                wraps=widget._refresh_composite_bounds,
            ) as refresh_bounds:
                with widget.labels_layer.block_history():
                    widget.labels_layer.data_setitem((rows, columns), 0)
                    # A repeated drag callback in the same stroke must not
                    # peel the newly revealed underlying class.
                    widget.labels_layer.data_setitem((rows, columns), 0)

            refresh_bounds.assert_called_once_with((0, 2, 0, 5))
            expected_top = np.array([1, 2, 1, 0], dtype=np.uint8)
            np.testing.assert_array_equal(
                widget.composite_layer.data[1, 1:5],
                expected_top,
            )
            self.assertEqual(store.memberships_at(1, 1), (1,))
            self.assertEqual(store.memberships_at(1, 2), (1, 2))
            self.assertEqual(store.memberships_at(1, 3), (1,))
            self.assertEqual(store.memberships_at(1, 4), ())
            self.assertEqual(store.memberships_at(0, 0), ())
            np.testing.assert_array_equal(widget.labels_layer.data, edit_before)

            tracker = widget.labels_layer._napari_histo_edit_tracker
            self.assertEqual(len(widget.labels_layer._undo_history), 1)
            self.assertEqual(len(tracker.undo_items), 1)
            packed_after = np.array(store.packed_masks, copy=True)
            projection_after = np.array(store.projection_view, copy=True)

            widget.labels_layer.undo()
            np.testing.assert_array_equal(store.packed_masks, packed_before)
            np.testing.assert_array_equal(
                store.projection_view,
                projection_before,
            )
            np.testing.assert_array_equal(widget.labels_layer.data, edit_before)
            self.assertEqual(len(tracker.redo_items), 1)

            widget.labels_layer.redo()
            np.testing.assert_array_equal(store.packed_masks, packed_after)
            np.testing.assert_array_equal(
                store.projection_view,
                projection_after,
            )
            np.testing.assert_array_equal(widget.labels_layer.data, edit_before)
            self.assertEqual(len(tracker.undo_items), 1)

    def test_semantic_eraser_pairs_active_and_other_tops_with_native_history(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            widget.labels_layer.data_setitem(
                (np.array([1]), np.array([2])),
                1,
            )
            widget._upsert_class(2, "B", "#123456")
            widget.labels_layer.data_setitem(
                (np.array([1]), np.array([2])),
                1,
            )
            widget._select_label(1)
            store = widget.overlap_store
            packed_before = np.array(store.packed_masks, copy=True)
            projection_before = np.array(store.projection_view, copy=True)
            edit_before = np.array(widget.labels_layer.data, copy=True)

            widget.labels_layer.data_setitem(
                (np.array([1, 1]), np.array([1, 2])),
                0,
            )

            np.testing.assert_array_equal(
                widget.composite_layer.data[1, 1:3],
                np.array([0, 1], dtype=np.uint8),
            )
            self.assertEqual(store.memberships_at(1, 1), ())
            self.assertEqual(store.memberships_at(1, 2), (1,))
            self.assertEqual(widget.labels_layer.data[1, 1], 0)
            self.assertEqual(widget.labels_layer.data[1, 2], 1)
            self.assertEqual(len(widget.labels_layer._undo_history), 1)

            packed_after = np.array(store.packed_masks, copy=True)
            projection_after = np.array(store.projection_view, copy=True)
            edit_after = np.array(widget.labels_layer.data, copy=True)
            widget.labels_layer.undo()
            np.testing.assert_array_equal(store.packed_masks, packed_before)
            np.testing.assert_array_equal(
                store.projection_view,
                projection_before,
            )
            np.testing.assert_array_equal(widget.labels_layer.data, edit_before)

            widget.labels_layer.redo()
            np.testing.assert_array_equal(store.packed_masks, packed_after)
            np.testing.assert_array_equal(
                store.projection_view,
                projection_after,
            )
            np.testing.assert_array_equal(widget.labels_layer.data, edit_after)

    def test_semantic_eraser_controller_failure_rolls_back_native_state(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            layer = widget.labels_layer
            tracker = layer._napari_histo_edit_tracker
            store = widget.overlap_store
            layer._reset_history()
            tracker.clear()
            binary_before = np.array(layer.data, copy=True)
            packed_before = np.array(store.packed_masks, copy=True)
            projection_before = np.array(store.projection_view, copy=True)
            hash_before = store.projection_sha256
            generation_before = store.generation
            native_before = (
                tuple(map(id, layer._undo_history)),
                tuple(map(id, layer._redo_history)),
                tuple(map(id, layer._staged_history)),
            )
            custom_before = (
                tuple(map(id, tracker.undo_items)),
                tuple(map(id, tracker.redo_items)),
                tuple(map(id, tracker.staged)),
            )

            with patch.object(
                widget.overlap_editor,
                "erase_visible_indices",
                side_effect=MemoryError("forced controller failure"),
            ):
                with self.assertRaisesRegex(MemoryError, "forced controller"):
                    layer.data_setitem(
                        (np.array([1]), np.array([1])),
                        0,
                    )

            np.testing.assert_array_equal(layer.data, binary_before)
            np.testing.assert_array_equal(store.packed_masks, packed_before)
            np.testing.assert_array_equal(
                store.projection_view,
                projection_before,
            )
            self.assertEqual(store.projection_sha256, hash_before)
            self.assertEqual(store.generation, generation_before)
            self.assertEqual(
                (
                    tuple(map(id, layer._undo_history)),
                    tuple(map(id, layer._redo_history)),
                    tuple(map(id, layer._staged_history)),
                ),
                native_before,
            )
            self.assertEqual(
                (
                    tuple(map(id, tracker.undo_items)),
                    tuple(map(id, tracker.redo_items)),
                    tuple(map(id, tracker.staged)),
                ),
                custom_before,
            )

    def test_failed_later_erase_callback_commits_prior_staged_action(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            layer = widget.labels_layer
            tracker = layer._napari_histo_edit_tracker
            layer.data_setitem(
                (np.array([1]), np.array([2])),
                1,
            )
            layer._reset_history()
            tracker.clear()
            native_erase = widget.overlap_editor.erase_visible_indices
            calls = 0

            def fail_second(indices):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise MemoryError("forced second callback failure")
                return native_erase(indices)

            with patch.object(
                widget.overlap_editor,
                "erase_visible_indices",
                side_effect=fail_second,
            ):
                with self.assertRaisesRegex(MemoryError, "second callback"):
                    with layer.block_history():
                        layer.data_setitem(
                            (np.array([1]), np.array([1])),
                            0,
                        )
                        layer.data_setitem(
                            (np.array([1]), np.array([2])),
                            0,
                        )

            self.assertEqual(widget.composite_layer.data[1, 1], 0)
            self.assertEqual(widget.composite_layer.data[1, 2], 1)
            self.assertEqual(layer.data[1, 1], 0)
            self.assertEqual(layer.data[1, 2], 1)
            self.assertFalse(layer._staged_history)
            self.assertFalse(tracker.staged)
            self.assertEqual(len(layer._undo_history), 1)
            self.assertEqual(len(tracker.undo_items), 1)

            layer.undo()
            self.assertEqual(widget.composite_layer.data[1, 1], 1)
            self.assertEqual(widget.composite_layer.data[1, 2], 1)
            self.assertEqual(layer.data[1, 1], 1)
            self.assertEqual(layer.data[1, 2], 1)

    def test_erase_seen_memory_failure_keeps_aligned_undoable_action(self):
        class FailingSet(set):
            def update(self, *args, **kwargs):
                raise MemoryError("forced seen-set failure")

        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            layer = widget.labels_layer
            tracker = layer._napari_histo_edit_tracker
            layer.data_setitem(
                (np.array([1]), np.array([2])),
                1,
            )
            layer._reset_history()
            tracker.clear()
            tracker.semantic_erase_seen = FailingSet()
            layer.mode = Mode.ERASE

            with layer.block_history():
                layer.data_setitem(
                    (np.array([1]), np.array([1])),
                    0,
                )
                self.assertTrue(tracker.semantic_erase_dedup_exhausted)
                # The safe fallback ends this stroke instead of risking a
                # second-level peel without retained dedup coordinates.
                layer.data_setitem(
                    (np.array([1]), np.array([2])),
                    0,
                )

            self.assertEqual(widget.composite_layer.data[1, 1], 0)
            self.assertEqual(widget.composite_layer.data[1, 2], 1)
            self.assertEqual(len(layer._undo_history), 1)
            self.assertEqual(len(tracker.undo_items), 1)
            layer.undo()
            self.assertEqual(widget.composite_layer.data[1, 1], 1)
            self.assertEqual(layer.data[1, 1], 1)

    def test_history_refresh_failure_completes_semantic_transition(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            layer = widget.labels_layer
            tracker = layer._napari_histo_edit_tracker
            store = widget.overlap_store
            layer.data_setitem(
                (np.array([1]), np.array([1])),
                0,
            )

            with patch.object(
                widget,
                "_refresh_composite_bounds",
                side_effect=MemoryError("forced display failure"),
            ):
                layer.undo()
            self.assertEqual(layer.data[1, 1], 1)
            self.assertEqual(store.memberships_at(1, 1), (1,))
            self.assertEqual(store.projection_view[1, 1], 1)
            self.assertEqual(len(layer._undo_history), 0)
            self.assertEqual(len(layer._redo_history), 1)
            self.assertEqual(len(tracker.undo_items), 0)
            self.assertEqual(len(tracker.redo_items), 1)
            self.assertIn("display could not refresh", widget.viewer.status)

            with patch.object(
                widget,
                "_refresh_composite_bounds",
                side_effect=MemoryError("forced display failure"),
            ):
                layer.redo()
            self.assertEqual(layer.data[1, 1], 0)
            self.assertEqual(store.memberships_at(1, 1), ())
            self.assertEqual(store.projection_view[1, 1], 0)
            self.assertEqual(len(layer._undo_history), 1)
            self.assertEqual(len(layer._redo_history), 0)
            self.assertEqual(len(tracker.undo_items), 1)
            self.assertEqual(len(tracker.redo_items), 0)

    def test_semantic_undo_redo_failure_is_an_exact_no_op(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            layer = widget.labels_layer
            tracker = layer._napari_histo_edit_tracker
            store = widget.overlap_store
            layer.data_setitem(
                (np.array([1]), np.array([2])),
                1,
            )
            layer._reset_history()
            tracker.clear()
            with layer.block_history():
                layer.data_setitem(
                    (np.array([1]), np.array([1])),
                    0,
                )
                layer.data_setitem(
                    (np.array([1]), np.array([2])),
                    0,
                )

            def snapshot():
                return {
                    "binary": np.array(layer.data, copy=True),
                    "packed": np.array(store.packed_masks, copy=True),
                    "projection": np.array(store.projection_view, copy=True),
                    "hash": store.projection_sha256,
                    "generation": store.generation,
                    "native_undo": tuple(map(id, layer._undo_history)),
                    "native_redo": tuple(map(id, layer._redo_history)),
                    "custom_undo": tuple(map(id, tracker.undo_items)),
                    "custom_redo": tuple(map(id, tracker.redo_items)),
                }

            def assert_snapshot(expected):
                np.testing.assert_array_equal(layer.data, expected["binary"])
                np.testing.assert_array_equal(
                    store.packed_masks,
                    expected["packed"],
                )
                np.testing.assert_array_equal(
                    store.projection_view,
                    expected["projection"],
                )
                self.assertEqual(store.projection_sha256, expected["hash"])
                self.assertEqual(store.generation, expected["generation"])
                self.assertEqual(
                    tuple(map(id, layer._undo_history)),
                    expected["native_undo"],
                )
                self.assertEqual(
                    tuple(map(id, layer._redo_history)),
                    expected["native_redo"],
                )
                self.assertEqual(
                    tuple(map(id, tracker.undo_items)),
                    expected["custom_undo"],
                )
                self.assertEqual(
                    tuple(map(id, tracker.redo_items)),
                    expected["custom_redo"],
                )

            def fail_second_restore(native_restore):
                calls = 0

                def restore(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls >= 2:
                        raise MemoryError("forced persistent history failure")
                    return native_restore(*args, **kwargs)

                return restore

            erased = snapshot()
            native_restore = widget.overlap_editor.restore_erased_indices
            with patch.object(
                widget.overlap_editor,
                "restore_erased_indices",
                side_effect=fail_second_restore(native_restore),
            ), self.assertRaisesRegex(MemoryError, "persistent history"):
                layer.undo()
            assert_snapshot(erased)

            layer.undo()
            restored = snapshot()
            with patch.object(
                widget.overlap_editor,
                "restore_erased_indices",
                side_effect=fail_second_restore(native_restore),
            ), self.assertRaisesRegex(MemoryError, "persistent history"):
                layer.redo()
            assert_snapshot(restored)

            layer.redo()
            np.testing.assert_array_equal(store.packed_masks, erased["packed"])
            np.testing.assert_array_equal(
                store.projection_view,
                erased["projection"],
            )

    def test_fill_erase_removes_visible_component_with_unrelated_active_class(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            rows, columns = np.indices((3, 3))
            class_one = (rows.reshape(-1) + 1, columns.reshape(-1) + 1)
            center = (np.array([2]), np.array([2]))
            widget.labels_layer.data_setitem(class_one, 1)
            widget._upsert_class(2, "B", "#123456")
            widget.labels_layer.data_setitem(center, 1)
            widget._upsert_class(25, "Unrelated", "#654321")
            self.assertFalse(np.any(widget.labels_layer.data))
            tracker = widget.labels_layer._napari_histo_edit_tracker
            widget.labels_layer.mode = Mode.FILL

            with widget.labels_layer.block_history():
                widget.labels_layer.fill((2, 2), 0)
                self.assertFalse(tracker.semantic_erase_seen)
                self.assertTrue(tracker.semantic_erase_compact_done)
                # After B is removed, the clicked A component expands from
                # one pixel to 3x3. A second callback in the same Fill action
                # must not peel that newly revealed underlying component.
                widget.labels_layer.fill((2, 2), 0)
                self.assertFalse(tracker.semantic_erase_seen)
                self.assertTrue(tracker.semantic_erase_compact_done)

            np.testing.assert_array_equal(
                widget.composite_layer.data[1:4, 1:4],
                np.ones((3, 3), dtype=np.uint8),
            )
            self.assertEqual(
                widget.overlap_store.memberships_at(2, 2),
                (1,),
            )
            with patch.object(
                widget.overlap_editor,
                "full_sync",
                side_effect=AssertionError(
                    "semantic Fill history scanned the full slide"
                ),
            ):
                widget.undo()
                self.assertEqual(widget.composite_layer.data[2, 2], 2)
                self.assertEqual(
                    widget.overlap_store.memberships_at(2, 2),
                    (1, 2),
                )
                self.assertEqual(widget.composite_layer.data[1, 1], 1)
                widget.labels_layer.redo()
                np.testing.assert_array_equal(
                    widget.composite_layer.data[1:4, 1:4],
                    np.ones((3, 3), dtype=np.uint8),
                )

    def test_pick_selects_connected_visible_region_and_activates_its_class(
        self,
    ):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            self.assertEqual(widget.delete_region_btn.text(), "Delete…")
            self.assertEqual(widget.clear_region_btn.text(), "Clear")
            for removed_control in (
                "selection_help_label",
                "selection_status_label",
                "use_selection_class_btn",
                "move_region_step",
                "move_region_up_btn",
                "move_region_down_btn",
                "move_region_left_btn",
                "move_region_right_btn",
            ):
                self.assertFalse(hasattr(widget, removed_control))
            widget._upsert_class(2, "B", "#123456")
            widget.labels_layer.data_setitem(
                (np.array([2, 2, 3]), np.array([2, 3, 2])),
                1,
            )
            widget._upsert_class(300, "Review", "#abcdef")
            widget.labels_layer.data_setitem(
                (np.array([5]), np.array([5])),
                1,
            )
            callback = widget.labels_layer._drag_modes[Mode.PICK]
            event = SimpleNamespace(
                position=(2, 2),
                view_direction=None,
                dims_displayed=(0, 1),
            )

            self.assertIsNot(
                widget.labels_layer._drag_modes,
                type(widget.labels_layer)._drag_modes,
            )
            callback(widget.labels_layer, event)
            self.assertEqual(widget.overlap_editor.active_class, 2)
            self.assertEqual(widget.labels_layer.selected_label, 1)
            self.assertEqual(widget._annotation_selection.value, 2)
            self.assertEqual(widget._annotation_selection.pixel_count, 3)
            self.assertTrue(widget._selection_layer.visible)
            self.assertFalse(widget._selection_layer.editable)
            self.assertTrue(widget.delete_region_btn.isEnabled())
            for path in widget._selection_layer.data:
                np.testing.assert_allclose(path[0], path[-1])

            widget._use_selected_annotation_class()
            self.assertEqual(widget.overlap_editor.active_class, 2)

            event.position = (5, 5)
            callback(widget.labels_layer, event)
            self.assertEqual(widget._annotation_selection.value, 300)
            self.assertEqual(widget.overlap_editor.active_class, 300)

            event.position = (0, 0)
            callback(widget.labels_layer, event)
            self.assertIsNone(widget._annotation_selection)
            self.assertFalse(widget._selection_layer.visible)
            self.assertFalse(widget.delete_region_btn.isEnabled())
            self.assertEqual(widget.overlap_editor.active_class, 300)

    def test_delete_connected_region_reveals_hidden_and_has_exact_history(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)
            rows = np.array([2, 2], dtype=np.intp)
            columns = np.array([2, 3], dtype=np.intp)
            widget.labels_layer.data_setitem((rows, columns), 1)
            widget._upsert_class(2, "Visible", "#123456")
            widget.labels_layer.data_setitem((rows, columns), 1)
            # Keep a deliberately unrelated class active. Deletion must erase
            # the visible class and pair an empty native Labels history atom.
            widget._upsert_class(3, "Unrelated", "#abcdef")
            self.assertEqual(widget.overlap_editor.active_class, 3)
            selection = self.pick_connected_region(widget, (2, 2))
            self.assertEqual(selection.value, 2)
            self.assertEqual(selection.pixel_count, 2)
            # Pick activates the clicked semantic class. Restore the
            # deliberately unrelated tool class for this delete-history case.
            widget._select_label(3)
            self.assertEqual(widget.overlap_editor.active_class, 3)
            self.assertFalse(np.any(widget.labels_layer.data))

            file_before = labels_path.read_bytes()
            packed_before = np.array(
                widget.overlap_store._packed_masks,
                copy=True,
            )
            projection_before = np.array(
                widget.overlap_editor.composite,
                copy=True,
            )
            tracker = widget.labels_layer._napari_histo_edit_tracker
            with patch.object(
                QMessageBox,
                "question",
                return_value=QMessageBox.Yes,
            ), patch.object(
                widget.overlap_editor,
                "full_sync",
                side_effect=AssertionError(
                    "connected deletion history scanned the full slide"
                ),
            ):
                widget._delete_selected_annotation()
                self.assertIsNone(widget._annotation_selection)
                np.testing.assert_array_equal(
                    widget.overlap_editor.composite[rows, columns],
                    np.ones(2, dtype=np.uint8),
                )
                for row, column in zip(rows, columns):
                    self.assertEqual(
                        widget.overlap_store.memberships_at(row, column),
                        (1,),
                    )
                self.assertEqual(labels_path.read_bytes(), file_before)
                self.assertEqual(len(widget.labels_layer._undo_history), 1)
                self.assertEqual(len(tracker.undo_items), 1)
                compact_atom = tracker.undo_items[-1][0]
                self.assertEqual(compact_atom[0][0].dtype, np.dtype(np.int32))
                self.assertEqual(compact_atom[0][1].dtype, np.dtype(np.int32))
                self.assertEqual(np.asarray(compact_atom[1]).ndim, 0)
                self.assertEqual(np.asarray(compact_atom[3]).ndim, 0)

                widget.undo()
                np.testing.assert_array_equal(
                    widget.overlap_store._packed_masks,
                    packed_before,
                )
                np.testing.assert_array_equal(
                    widget.overlap_editor.composite,
                    projection_before,
                )
                for row, column in zip(rows, columns):
                    self.assertEqual(
                        set(widget.overlap_store.memberships_at(row, column)),
                        {1, 2},
                    )
                self.assertEqual(len(widget.labels_layer._redo_history), 1)
                self.assertEqual(len(tracker.redo_items), 1)

                widget.labels_layer.redo()
                np.testing.assert_array_equal(
                    widget.overlap_editor.composite[rows, columns],
                    np.ones(2, dtype=np.uint8),
                )
                for row, column in zip(rows, columns):
                    self.assertEqual(
                        widget.overlap_store.memberships_at(row, column),
                        (1,),
                    )
                self.assertEqual(len(widget.labels_layer._undo_history), 1)
                self.assertEqual(len(tracker.undo_items), 1)
                self.assertEqual(labels_path.read_bytes(), file_before)

    def test_selected_delete_failure_is_exact_noop_and_keeps_selection(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            selection = self.pick_connected_region(widget, (1, 1))
            state_before = self.connected_runtime_state(widget)

            with patch.object(
                QMessageBox,
                "question",
                return_value=QMessageBox.Yes,
            ), patch.object(
                widget.overlap_store,
                "_set_membership_at_indices",
                side_effect=MemoryError("forced selected erase failure"),
            ):
                widget._delete_selected_annotation()

            self.assertEqual(self.connected_runtime_state(widget), state_before)
            self.assertIs(widget._annotation_selection, selection)
            self.assertTrue(widget.delete_region_btn.isEnabled())
            self.assertIn("Could not delete", widget.viewer.status)

    def test_selected_delete_refresh_failure_is_committed_and_truthful(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            selection = self.pick_connected_region(widget, (1, 1))
            rows = selection.rows.copy()
            columns = selection.columns.copy()
            tracker = widget.labels_layer._napari_histo_edit_tracker

            with patch.object(
                QMessageBox,
                "question",
                return_value=QMessageBox.Yes,
            ), patch.object(
                widget,
                "_refresh_composite_bounds",
                side_effect=RuntimeError("forced display upload failure"),
            ):
                widget._delete_selected_annotation()

            self.assertIsNone(widget._annotation_selection)
            self.assertFalse(widget.delete_region_btn.isEnabled())
            np.testing.assert_array_equal(
                widget.overlap_editor.composite[rows, columns],
                np.zeros(rows.shape, dtype=widget.overlap_editor.composite.dtype),
            )
            self.assertEqual(len(widget.labels_layer._undo_history), 1)
            self.assertEqual(len(tracker.undo_items), 1)
            self.assertIn("deleted in memory", widget.viewer.status)
            self.assertIn("display could not refresh", widget.viewer.status)

    def test_selected_delete_history_never_corrupts_switched_active_proxy(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            widget._upsert_class(2, "Other", "#123456")
            widget.labels_layer.data_setitem(
                (np.array([4]), np.array([4])),
                1,
            )
            selection = self.pick_connected_region(widget, (1, 1))
            self.assertEqual(selection.value, 1)

            with patch.object(
                QMessageBox,
                "question",
                return_value=QMessageBox.Yes,
            ):
                widget._delete_selected_annotation()

            # Simulate a retained-history class switch without the ordinary
            # UI helper's deliberate history reset (for example an external
            # controller caller). Native history must still never target this
            # newly active binary proxy with the deleted class's coordinates.
            widget.overlap_editor.select_class(2)
            expected_proxy = widget.overlap_store.select_plane(2).astype(
                np.uint8
            )
            np.testing.assert_array_equal(widget.labels_layer.data, expected_proxy)

            widget.undo()
            np.testing.assert_array_equal(widget.labels_layer.data, expected_proxy)
            np.testing.assert_array_equal(
                widget.overlap_editor.composite[
                    selection.rows,
                    selection.columns,
                ],
                np.ones(selection.rows.shape, dtype=np.uint8),
            )

            widget.labels_layer.redo()
            np.testing.assert_array_equal(widget.labels_layer.data, expected_proxy)
            np.testing.assert_array_equal(
                widget.overlap_editor.composite[
                    selection.rows,
                    selection.columns,
                ],
                np.zeros(selection.rows.shape, dtype=np.uint8),
            )

    def test_move_connected_overlap_is_one_sparse_undo_with_unrelated_active(
        self,
    ):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)
            row = np.full(4, 2, dtype=np.intp)
            base_columns = np.arange(2, 6, dtype=np.intp)
            widget.labels_layer.data_setitem((row, base_columns), 1)

            widget._upsert_class(2, "Moving", "#123456")
            source_columns = np.arange(2, 5, dtype=np.intp)
            source_rows = np.full(3, 2, dtype=np.intp)
            widget.labels_layer.data_setitem(
                (source_rows, source_columns),
                1,
            )
            widget._upsert_class(3, "Destination top", "#654321")
            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([5])),
                1,
            )
            widget._upsert_class(4, "Unrelated active", "#abcdef")
            self.assertEqual(widget.overlap_editor.active_class, 4)
            self.assertFalse(np.any(widget.labels_layer.data))

            selection = self.pick_connected_region(widget, (2, 2))
            self.assertEqual(selection.value, 2)
            self.assertEqual(selection.pixel_count, 3)
            # Pick activates class 2 by design; this test specifically covers
            # sparse move history when a different class is active.
            widget._select_label(4)
            self.assertEqual(widget.overlap_editor.active_class, 4)
            self.assertFalse(np.any(widget.labels_layer.data))
            packed_before = np.array(
                widget.overlap_store._packed_masks,
                copy=True,
            )
            projection_before = np.array(
                widget.overlap_editor.composite,
                copy=True,
            )
            proxy_before = np.array(widget.labels_layer.data, copy=True)
            file_before = labels_path.read_bytes()
            tracker = widget.labels_layer._napari_histo_edit_tracker

            with patch.object(
                widget.overlap_editor,
                "full_sync",
                side_effect=AssertionError("move history scanned the slide"),
            ), patch.object(
                widget.overlap_store,
                "project",
                side_effect=AssertionError("move rebuilt the projection"),
            ), patch.object(
                widget.overlap_store,
                "select_plane",
                side_effect=AssertionError("move unpacked a class plane"),
            ):
                widget._move_selected_annotation(0, 1)
                np.testing.assert_array_equal(
                    widget.overlap_editor.composite[2, 2:6],
                    np.array([1, 2, 2, 2], dtype=np.uint8),
                )
                self.assertEqual(
                    widget.overlap_store.memberships_at(2, 2),
                    (1,),
                )
                self.assertEqual(
                    set(widget.overlap_store.memberships_at(2, 5)),
                    {1, 2, 3},
                )
                self.assertEqual(
                    widget.overlap_store.memberships_at(2, 5)[-1],
                    2,
                )
                self.assertEqual(widget.overlap_editor.active_class, 4)
                np.testing.assert_array_equal(
                    widget.labels_layer.data,
                    proxy_before,
                )
                self.assertEqual(len(widget.labels_layer._undo_history), 1)
                self.assertEqual(len(tracker.undo_items), 1)
                packed_after = np.array(
                    widget.overlap_store._packed_masks,
                    copy=True,
                )
                projection_after = np.array(
                    widget.overlap_editor.composite,
                    copy=True,
                )

                widget.undo()
                np.testing.assert_array_equal(
                    widget.overlap_store._packed_masks,
                    packed_before,
                )
                np.testing.assert_array_equal(
                    widget.overlap_editor.composite,
                    projection_before,
                )
                np.testing.assert_array_equal(
                    widget.labels_layer.data,
                    proxy_before,
                )
                self.assertEqual(len(widget.labels_layer._undo_history), 0)
                self.assertEqual(len(widget.labels_layer._redo_history), 1)
                self.assertEqual(len(tracker.undo_items), 0)
                self.assertEqual(len(tracker.redo_items), 1)

                widget.labels_layer.redo()
                np.testing.assert_array_equal(
                    widget.overlap_store._packed_masks,
                    packed_after,
                )
                np.testing.assert_array_equal(
                    widget.overlap_editor.composite,
                    projection_after,
                )
                np.testing.assert_array_equal(
                    widget.labels_layer.data,
                    proxy_before,
                )
                self.assertEqual(len(widget.labels_layer._undo_history), 1)
                self.assertEqual(len(tracker.undo_items), 1)
                self.assertEqual(widget.overlap_editor.active_class, 4)
            self.assertEqual(labels_path.read_bytes(), file_before)

    def test_move_connected_active_class_keeps_native_history_aligned(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            source_rows = np.array([3, 3], dtype=np.intp)
            source_columns = np.array([2, 3], dtype=np.intp)
            widget.labels_layer.data_setitem(
                (source_rows, source_columns),
                1,
            )
            widget._limit_undo_history(widget.labels_layer)
            selection = self.pick_connected_region(widget, (3, 2))
            self.assertEqual(selection.value, 1)
            self.assertEqual(widget.overlap_editor.active_class, 1)
            packed_before = np.array(
                widget.overlap_store._packed_masks,
                copy=True,
            )
            projection_before = np.array(
                widget.overlap_editor.composite,
                copy=True,
            )
            proxy_before = np.array(widget.labels_layer.data, copy=True)
            tracker = widget.labels_layer._napari_histo_edit_tracker

            with patch.object(
                widget.overlap_editor,
                "full_sync",
                side_effect=AssertionError("active move scanned the slide"),
            ), patch.object(
                widget.overlap_store,
                "project",
                side_effect=AssertionError("active move rebuilt projection"),
            ), patch.object(
                widget.overlap_store,
                "select_plane",
                side_effect=AssertionError("active move unpacked a plane"),
            ):
                widget._move_selected_annotation(0, 1)
                np.testing.assert_array_equal(
                    widget.labels_layer.data[3, 2:5],
                    np.array([0, 1, 1], dtype=np.uint8),
                )
                packed_after = np.array(
                    widget.overlap_store._packed_masks,
                    copy=True,
                )
                projection_after = np.array(
                    widget.overlap_editor.composite,
                    copy=True,
                )
                proxy_after = np.array(widget.labels_layer.data, copy=True)
                self.assertEqual(len(widget.labels_layer._undo_history), 1)
                self.assertEqual(len(tracker.undo_items), 1)

                widget.undo()
                np.testing.assert_array_equal(
                    widget.overlap_store._packed_masks,
                    packed_before,
                )
                np.testing.assert_array_equal(
                    widget.overlap_editor.composite,
                    projection_before,
                )
                np.testing.assert_array_equal(
                    widget.labels_layer.data,
                    proxy_before,
                )
                self.assertEqual(len(widget.labels_layer._redo_history), 1)
                self.assertEqual(len(tracker.redo_items), 1)

                widget.labels_layer.redo()
                np.testing.assert_array_equal(
                    widget.overlap_store._packed_masks,
                    packed_after,
                )
                np.testing.assert_array_equal(
                    widget.overlap_editor.composite,
                    projection_after,
                )
                np.testing.assert_array_equal(
                    widget.labels_layer.data,
                    proxy_after,
                )
                self.assertEqual(len(widget.labels_layer._undo_history), 1)
                self.assertEqual(len(tracker.undo_items), 1)

    def test_connected_region_invalid_moves_are_exact_noops(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            selection = self.pick_connected_region(widget, (1, 1))
            self.assertIsNotNone(selection)

            state_before = self.connected_runtime_state(widget)
            widget._move_selected_annotation(-1, 0, step=2)
            self.assertEqual(
                self.connected_runtime_state(widget),
                state_before,
            )
            self.assertIn("outside the image", widget.viewer.status)

            stale = type(selection)(
                value=selection.value,
                rows=selection.rows,
                columns=selection.columns,
                outlines=selection.outlines,
                simplified_preview=selection.simplified_preview,
                store_revision=int(selection.store_revision) - 1,
            )
            widget._annotation_selection = stale
            stale_before = self.connected_runtime_state(widget)
            widget._move_selected_annotation(0, 1)
            self.assertEqual(
                self.connected_runtime_state(widget),
                stale_before,
            )
            self.assertIn("stale", widget.viewer.status)

            widget._set_annotation_selection(selection)
            widget._save_worker = object()
            widget._update_project_controls()
            try:
                saving_before = self.connected_runtime_state(widget)
                widget._move_selected_annotation(0, 1)
                self.assertEqual(
                    self.connected_runtime_state(widget),
                    saving_before,
                )
                self.assertIn("save to finish", widget.viewer.status)
                self.assertFalse(widget.delete_region_btn.isEnabled())
                self.assertFalse(widget.clear_region_btn.isEnabled())
            finally:
                widget._save_worker = None
                widget._update_project_controls()

    def test_hidden_or_removed_connected_outline_disables_actions(self):
        with TemporaryDirectory() as tmp:
            viewer, widget, _, _ = self.load_small_project(Path(tmp))
            self.assertIsNotNone(self.pick_connected_region(widget, (1, 1)))
            action_widgets = (
                widget.delete_region_btn,
                widget.clear_region_btn,
            )
            self.assertTrue(all(action.isEnabled() for action in action_widgets))

            outline = widget._selection_layer
            # napari Shapes can reset editable state while changing displayed
            # dimensions; the preview must never become a data-editing layer.
            outline.editable = True
            widget._lock_selection_preview_layer()
            self.assertFalse(outline.editable)
            outline.visible = False
            self.assertIsNotNone(widget._annotation_selection)
            self.assertTrue(all(not action.isEnabled() for action in action_widgets))
            self.assertIn("outline is hidden", widget.viewer.status)

            outline.visible = True
            self.assertTrue(all(action.isEnabled() for action in action_widgets))
            viewer.layers.remove(outline)
            self.assertIsNone(widget._selection_layer)
            self.assertIsNone(widget._annotation_selection)
            self.assertTrue(all(not action.isEnabled() for action in action_widgets))

    def test_entering_3d_clears_connected_selection_without_data_changes(self):
        with TemporaryDirectory() as tmp:
            viewer, widget, _, _ = self.load_small_project(Path(tmp))
            self.assertIsNotNone(self.pick_connected_region(widget, (1, 1)))
            outline = widget._selection_layer
            state_before = self.connected_runtime_state(widget)

            viewer.dims.ndisplay = 3

            self.assertEqual(viewer.dims.ndisplay, 3)
            self.assertEqual(
                self.connected_runtime_state(widget)[:10],
                state_before[:10],
            )
            self.assertIsNone(widget._annotation_selection)
            self.assertFalse(outline.visible)
            self.assertEqual(outline.data, [])
            action_widgets = (
                widget.delete_region_btn,
                widget.clear_region_btn,
            )
            self.assertTrue(all(not action.isEnabled() for action in action_widgets))

    def test_clear_outline_restores_annotation_tools_as_active_layer(self):
        with TemporaryDirectory() as tmp:
            viewer, widget, _, _ = self.load_small_project(Path(tmp))
            self.assertIsNotNone(self.pick_connected_region(widget, (1, 1)))
            outline = widget._selection_layer
            viewer.layers.selection.active = outline
            self.assertIs(viewer.layers.selection.active, outline)
            state_before = self.connected_runtime_state(widget)

            widget.clear_region_btn.click()

            self.assertEqual(
                self.connected_runtime_state(widget)[:10],
                state_before[:10],
            )
            self.assertIsNone(widget._annotation_selection)
            self.assertFalse(outline.visible)
            self.assertEqual(outline.data, [])
            self.assertIs(viewer.layers.selection.active, widget.labels_layer)
            self.assertFalse(widget.delete_region_btn.isEnabled())
            self.assertFalse(widget.clear_region_btn.isEnabled())

    def test_failed_replacement_outline_never_leaves_old_selection_armed(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            widget.labels_layer.data_setitem(
                (np.array([4]), np.array([4])),
                1,
            )
            selected_a = self.pick_connected_region(widget, (1, 1))
            self.assertIsNotNone(selected_a)
            self.assertTrue(widget.delete_region_btn.isEnabled())
            state_before = self.connected_runtime_state(widget)

            with patch.object(
                widget._selection_layer,
                "add",
                side_effect=RuntimeError("injected outline failure"),
            ):
                self.pick_connected_region(widget, (4, 4))

            # The failed preview may clear UI-only outline state, but it must
            # not touch membership, projection, proxy, or either history pair.
            self.assertEqual(
                self.connected_runtime_state(widget)[:10],
                state_before[:10],
            )
            self.assertIsNone(widget._annotation_selection)
            self.assertFalse(widget._selection_layer.visible)
            action_widgets = (
                widget.delete_region_btn,
                widget.clear_region_btn,
            )
            self.assertTrue(all(not action.isEnabled() for action in action_widgets))
            self.assertIn("nothing is selected", widget.viewer.status)

    def test_delete_confirmation_that_starts_save_is_exact_noop(self):
        with TemporaryDirectory() as tmp:
            _, widget, labels_path, _ = self.load_small_project(Path(tmp))
            selection = self.pick_connected_region(widget, (1, 1))
            self.assertIsNotNone(selection)
            state_before = self.connected_runtime_state(widget)
            file_before = labels_path.read_bytes()
            worker = FakeSaveWorker()

            def start_save_then_confirm(*args, **kwargs):
                del args, kwargs
                widget.save_labels()
                self.assertIs(widget._save_worker, worker)
                return QMessageBox.Yes

            with patch(
                "napari_histo_label_editor._widget.create_worker",
                return_value=worker,
            ), patch.object(
                QMessageBox,
                "question",
                side_effect=start_save_then_confirm,
            ):
                widget._delete_selected_annotation()

            self.assertEqual(worker.start_count, 1)
            self.assertIs(widget._save_worker, worker)
            self.assertEqual(self.connected_runtime_state(widget), state_before)
            self.assertIs(widget._annotation_selection, selection)
            self.assertEqual(labels_path.read_bytes(), file_before)
            self.assertIn("save to finish before deleting", widget.viewer.status)
            worker.finished.emit()

    def test_delete_confirmation_selection_change_modifies_neither_object(self):
        with TemporaryDirectory() as tmp:
            _, widget, labels_path, _ = self.load_small_project(Path(tmp))
            widget._upsert_class(2, "Second", "#123456")
            widget.labels_layer.data_setitem(
                (np.array([4]), np.array([4])),
                1,
            )
            selected_a = self.pick_connected_region(widget, (1, 1))
            self.assertEqual(selected_a.value, 1)
            state_before = self.connected_runtime_state(widget)
            file_before = labels_path.read_bytes()

            def select_b_then_confirm(*args, **kwargs):
                del args, kwargs
                selected_b = self.pick_connected_region(widget, (4, 4))
                self.assertEqual(selected_b.value, 2)
                return QMessageBox.Yes

            with patch.object(
                QMessageBox,
                "question",
                side_effect=select_b_then_confirm,
            ):
                widget._delete_selected_annotation()

            state_after = self.connected_runtime_state(widget)
            # Pick changed only the active tool class/binary proxy. The
            # guarded delete must leave authoritative memberships, projection,
            # revision/generation/hash, and both history pairs untouched.
            self.assertEqual(state_after[:2], state_before[:2])
            self.assertEqual(state_after[3:10], state_before[3:10])
            self.assertIsNot(widget._annotation_selection, selected_a)
            self.assertEqual(widget._annotation_selection.value, 2)
            self.assertEqual(widget.overlap_editor.active_class, 2)
            self.assertEqual(widget.labels_layer.data[4, 4], 1)
            self.assertEqual(np.count_nonzero(widget.labels_layer.data), 1)
            self.assertEqual(widget.overlap_editor.composite[1, 1], 1)
            self.assertEqual(widget.overlap_editor.composite[4, 4], 2)
            self.assertEqual(
                widget.overlap_store.memberships_at(1, 1),
                (1,),
            )
            self.assertEqual(
                widget.overlap_store.memberships_at(4, 4),
                (2,),
            )
            self.assertEqual(labels_path.read_bytes(), file_before)
            self.assertIn("changed while confirming", widget.viewer.status)

    def test_paired_history_destination_append_failure_is_exact_noop(self):
        class FailingAppendDeque(deque):
            def append(self, item):
                del item
                raise MemoryError("injected paired-history allocation failure")

        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            widget.labels_layer.data_setitem(
                (np.array([3]), np.array([4])),
                1,
            )
            tracker = widget.labels_layer._napari_histo_edit_tracker
            self.assertEqual(len(tracker.undo_items), 1)

            tracker.redo_items = FailingAppendDeque(
                tracker.redo_items,
                maxlen=tracker.redo_items.maxlen,
            )
            undo_before = self.connected_runtime_state(widget)
            with self.assertRaisesRegex(
                MemoryError,
                "paired-history allocation failure",
            ):
                widget.labels_layer.undo()
            self.assertEqual(self.connected_runtime_state(widget), undo_before)

            tracker.redo_items = deque(
                tracker.redo_items,
                maxlen=tracker.redo_items.maxlen,
            )
            widget.labels_layer.undo()
            self.assertEqual(len(tracker.redo_items), 1)
            tracker.undo_items = FailingAppendDeque(
                tracker.undo_items,
                maxlen=tracker.undo_items.maxlen,
            )
            redo_before = self.connected_runtime_state(widget)
            with self.assertRaisesRegex(
                MemoryError,
                "paired-history allocation failure",
            ):
                widget.labels_layer.redo()
            self.assertEqual(self.connected_runtime_state(widget), redo_before)

    def test_move_refresh_failure_keeps_applied_move_and_history(self):
        with TemporaryDirectory() as tmp:
            _, widget, _, _ = self.load_small_project(Path(tmp))
            widget.labels_layer.data_setitem(
                (np.array([2, 2]), np.array([2, 3])),
                1,
            )
            widget._limit_undo_history(widget.labels_layer)
            selection = self.pick_connected_region(widget, (2, 2))
            packed_before = np.array(
                widget.overlap_store._packed_masks,
                copy=True,
            )
            projection_before = np.array(
                widget.overlap_editor.composite,
                copy=True,
            )
            tracker = widget.labels_layer._napari_histo_edit_tracker
            with patch.object(
                widget,
                "_refresh_composite_bounds",
                side_effect=MemoryError("injected display upload failure"),
            ):
                widget._move_selected_annotation(0, 1)

            np.testing.assert_array_equal(
                widget.overlap_editor.composite[2, 2:5],
                np.array([0, 1, 1], dtype=np.uint8),
            )
            self.assertEqual(len(widget.labels_layer._undo_history), 1)
            self.assertEqual(len(tracker.undo_items), 1)
            self.assertNotIn("Could not move", widget.viewer.status)
            self.assertIsNotNone(widget._annotation_selection)
            np.testing.assert_array_equal(
                widget._annotation_selection.columns,
                selection.columns + 1,
            )

            widget.undo()
            np.testing.assert_array_equal(
                widget.overlap_store._packed_masks,
                packed_before,
            )
            np.testing.assert_array_equal(
                widget.overlap_editor.composite,
                projection_before,
            )

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
            widget.save_destination_line.setText(str(rescued_path))

            with patch(
                "napari_histo_label_editor._widget.QFileDialog.getSaveFileName",
            ) as save_as_dialog, patch(
                "napari_histo_label_editor._widget.atomic_write_class_config"
            ) as write_class_config:
                widget.save_labels()
                self.wait_for_save(widget)

            save_as_dialog.assert_not_called()
            write_class_config.assert_not_called()
            self.assertEqual(labels_path.read_bytes(), label_bytes_before)
            self.assertEqual(mapping_path.read_bytes(), external_mapping_bytes)
            self.assertTrue(rescued_path.is_file())
            self.assertEqual(iio.imread(rescued_path)[1, 1], 0)
            self.assertIsNotNone(read_embedded_annotations(rescued_path))
            self.assertEqual(widget.labels_path, rescued_path.resolve())
            self.assertEqual(widget.label_line.text(), str(labels_path.resolve()))
            self.assertEqual(widget._loaded_labels_path, labels_path.resolve())
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
