import unittest
from unittest.mock import patch

from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication, QDialog, QDialogButtonBox

from napari_histo_label_editor._class_dialog import ClassEditorDialog


class ClassEditorDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_add_dialog_exposes_normalized_values(self):
        dialog = ClassEditorDialog(
            3, "  Tumor  ", "#AABBCC", editing=False
        )

        self.assertFalse(dialog.editing)
        self.assertTrue(dialog.value_spin.isEnabled())
        self.assertEqual(dialog.values(), (3, "Tumor", "#aabbcc"))

        dialog.value = 4
        dialog.name = "Stroma"
        dialog.color = "red"
        self.assertEqual(dialog.values(), (4, "Stroma", "#ff0000"))

    def test_edit_dialog_locks_class_value(self):
        dialog = ClassEditorDialog(7, "Immune", "#123456", editing=True)

        self.assertTrue(dialog.editing)
        self.assertFalse(dialog.value_spin.isEnabled())
        self.assertEqual(dialog.value, 7)

    def test_blank_name_disables_ok_and_cannot_accept(self):
        dialog = ClassEditorDialog(2, "  ", "#123456", editing=False)
        ok_button = dialog.button_box.button(QDialogButtonBox.Ok)

        self.assertFalse(ok_button.isEnabled())
        dialog.accept()
        self.assertNotEqual(dialog.result(), QDialog.Accepted)

        dialog.name = "Epithelium"
        self.assertTrue(ok_button.isEnabled())
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.Accepted)

    def test_color_button_uses_native_color_dialog(self):
        dialog = ClassEditorDialog(2, "Epithelium", "#123456", editing=False)

        with patch(
            "napari_histo_label_editor._class_dialog.QColorDialog.getColor",
            return_value=QColor("#abcdef"),
        ):
            dialog.color_button.click()

        self.assertEqual(dialog.color, "#abcdef")
        self.assertEqual(dialog.color_button.text(), "#abcdef")

    def test_invalid_initial_or_assigned_color_is_rejected(self):
        with self.assertRaises(ValueError):
            ClassEditorDialog(2, "Epithelium", "not-a-color", editing=False)

        dialog = ClassEditorDialog(2, "Epithelium", "#123456", editing=False)
        with self.assertRaises(ValueError):
            dialog.color = "not-a-color"


if __name__ == "__main__":
    unittest.main()
