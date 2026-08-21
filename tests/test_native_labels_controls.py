import unittest

import numpy as np
from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari.layers import Labels
from napari.layers.labels._labels_constants import Mode
from qtpy.QtWidgets import QApplication

from napari_histo_label_editor._native_labels_controls import (
    adapt_native_labels_tool_controls,
)


class NativeLabelsToolControlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.layer = Labels(np.zeros((8, 8), dtype=np.uint8))
        self.controls = QtLabelsControls(self.layer)

    def tearDown(self):
        self.controls.close()

    def test_large_range_is_visible_and_shared_by_paint_and_erase(self):
        slider = self.controls._brush_size_slider_control.brush_size_slider

        self.assertEqual(slider.maximum(), 40)
        self.assertTrue(adapt_native_labels_tool_controls(self.controls))
        self.assertEqual(slider.minimum(), 1)
        self.assertEqual(slider.maximum(), 512)
        self.assertEqual(slider.singleStep(), 1)
        self.assertEqual(slider.pageStep(), 25)
        self.assertIn("Paint/Erase", slider.toolTip())
        self.assertIn("1–512", slider.toolTip())

        expected_mode = slider.EdgeLabelMode.LabelIsValue
        self.assertEqual(slider.edgeLabelMode(), expected_mode)
        self.assertEqual(
            self.controls._brush_size_slider_control.brush_size_slider_label.text(),
            "brush size 1–512:",
        )
        slider.setValue(401)
        self.assertEqual(self.layer.brush_size, 401)
        self.assertEqual(slider._label.text(), "401")

    def test_eraser_uses_brush_icon_without_changing_behavior_or_help(self):
        button = self.controls.erase_button
        original_tooltip = button.toolTip()
        original_mode = button.mode

        self.assertTrue(adapt_native_labels_tool_controls(self.controls))
        self.assertEqual(button.property("mode"), "paint")
        self.assertEqual(button.mode, original_mode)
        self.assertEqual(button.mode, Mode.ERASE)
        self.assertEqual(button.toolTip(), original_tooltip)

        # Toggling exercises QtModeRadioButton's real layer-mode connection.
        # A standalone controls widget is intentionally not clicked here:
        # napari's global action injection expects a containing Viewer.
        button.setChecked(True)
        self.assertEqual(self.layer.mode, Mode.ERASE)

    def test_adaptation_is_idempotent_and_preserves_larger_existing_size(self):
        self.layer.brush_size = 700

        self.assertTrue(adapt_native_labels_tool_controls(self.controls))
        self.assertTrue(adapt_native_labels_tool_controls(self.controls))
        slider = self.controls._brush_size_slider_control.brush_size_slider
        self.assertEqual(slider.maximum(), 700)
        self.assertEqual(slider.value(), 700)
        self.assertEqual(self.layer.brush_size, 700)

    def test_missing_or_invalid_controls_are_safe_to_retry(self):
        self.assertFalse(adapt_native_labels_tool_controls(None))
        self.assertFalse(
            adapt_native_labels_tool_controls(
                self.controls,
                maximum_brush_size=0,
            )
        )


if __name__ == "__main__":
    unittest.main()
