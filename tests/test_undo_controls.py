import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import imageio.v3 as iio
import numpy as np
import pandas as pd
from napari.components import ViewerModel
from napari.layers.labels._labels_constants import Mode
from qtpy.QtWidgets import QApplication

from napari_histo_label_editor._widget import LabelEditorWidget


class UndoControlsRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _load_project(root: Path):
        image_path = root / "image.png"
        labels_path = root / "labels.tif"
        mapping_path = root / "mapping.csv"
        iio.imwrite(image_path, np.zeros((8, 10, 3), dtype=np.uint8))
        labels = np.zeros((8, 10), dtype=np.uint8)
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
        return viewer, widget

    @staticmethod
    def _pick(widget, position):
        callback = widget.labels_layer._drag_modes[Mode.PICK]
        event = type(
            "PickEvent",
            (),
            {
                "position": position,
                "view_direction": None,
                "dims_displayed": (0, 1),
            },
        )()
        callback(widget.labels_layer, event)

    def test_same_class_after_eraser_keeps_button_undo_with_outline_active(self):
        with TemporaryDirectory() as tmp:
            viewer, widget = self._load_project(Path(tmp))
            point = (np.array([2]), np.array([3]))
            widget.labels_layer.data_setitem(point, 1)
            tracker = widget.labels_layer._napari_histo_edit_tracker
            self.assertEqual(len(widget.labels_layer._undo_history), 1)
            self.assertEqual(len(tracker.undo_items), 1)

            widget._select_label(0)
            widget._select_label(1)
            self.assertEqual(len(widget.labels_layer._undo_history), 1)
            self.assertEqual(len(tracker.undo_items), 1)

            self._pick(widget, (2, 3))
            viewer.layers.selection.active = widget._selection_layer
            self.assertTrue(widget.undo_btn.isEnabled())
            observed_progress = []
            native_undo = widget._undo_labels_layer

            def observe_then_undo(layer):
                observed_progress.append(
                    (
                        widget.undo_btn.text(),
                        widget.undo_btn.isEnabled(),
                        widget.viewer.status,
                    )
                )
                return native_undo(layer)

            with patch.object(
                widget,
                "_undo_labels_layer",
                side_effect=observe_then_undo,
            ):
                widget.undo_btn.click()

            self.assertEqual(widget.overlap_editor.composite[2, 3], 0)
            self.assertEqual(widget.labels_layer.data[2, 3], 0)
            self.assertIsNone(widget._annotation_selection)
            self.assertIs(viewer.layers.selection.active, widget.labels_layer)
            self.assertEqual(widget.viewer.status, "Undo complete.")
            self.assertEqual(widget.undo_btn.text(), "Undo complete ✓")
            self.assertTrue(widget.undo_btn.isEnabled())
            self.assertEqual(
                observed_progress,
                [
                    (
                        "Undoing…",
                        False,
                        "Undoing the last annotation change…",
                    )
                ],
            )

    def test_u_hotkey_calls_the_same_working_undo_path(self):
        with TemporaryDirectory() as tmp:
            viewer, widget = self._load_project(Path(tmp))
            point = (np.array([3]), np.array([4]))
            widget.labels_layer.data_setitem(point, 1)
            widget._select_label(0)
            widget._select_label(1)

            undo_hotkey = next(
                callback
                for binding, callback in viewer.keymap.items()
                if str(binding) == "U"
            )
            undo_hotkey(viewer)

            self.assertEqual(widget.overlap_editor.composite[3, 4], 0)
            self.assertEqual(widget.labels_layer.data[3, 4], 0)
            self.assertEqual(widget.viewer.status, "Undo complete.")
            self.assertEqual(widget.undo_btn.text(), "Undo complete ✓")

    def test_no_history_is_visible_on_button_and_status(self):
        with TemporaryDirectory() as tmp:
            _, widget = self._load_project(Path(tmp))

            widget.undo_btn.click()

            self.assertEqual(widget.undo_btn.text(), "Nothing to undo")
            self.assertEqual(widget.viewer.status, "Nothing to undo.")
            self.assertTrue(widget.undo_btn.isEnabled())

    def test_reentrant_undo_request_is_ignored_safely(self):
        with TemporaryDirectory() as tmp:
            _, widget = self._load_project(Path(tmp))
            point = (np.array([4]), np.array([5]))
            widget.labels_layer.data_setitem(point, 1)
            native_undo = widget._undo_labels_layer
            calls = []

            def request_nested_undo_then_continue(layer):
                calls.append("outer")
                widget.undo()
                return native_undo(layer)

            with patch.object(
                widget,
                "_undo_labels_layer",
                side_effect=request_nested_undo_then_continue,
            ):
                widget.undo()

            self.assertEqual(calls, ["outer"])
            self.assertEqual(widget.overlap_editor.composite[4, 5], 0)
            self.assertEqual(widget.labels_layer.data[4, 5], 0)
            self.assertEqual(widget.undo_btn.text(), "Undo complete ✓")
            self.assertEqual(widget.viewer.status, "Undo complete.")


if __name__ == "__main__":
    unittest.main()
