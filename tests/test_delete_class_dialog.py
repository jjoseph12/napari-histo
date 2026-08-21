import unittest

from qtpy.QtWidgets import QApplication, QDialog

from napari_histo_label_editor._delete_class_dialog import DeleteClassDialog


class DeleteClassDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_shows_deleted_class_usage_and_replacement_choices(self):
        dialog = DeleteClassDialog(
            7,
            "Tumor",
            12_345,
            {0: "Background", 2: "Stroma", 7: "Tumor", 9: "Immune"},
        )

        self.assertIn("Tumor", dialog.class_label.text())
        self.assertIn("7", dialog.class_label.text())
        self.assertEqual(dialog.pixel_count_label.text(), "12,345 pixels")
        self.assertEqual(dialog.replacement_combo.count(), 3)
        self.assertEqual(
            {
                int(dialog.replacement_combo.itemData(index))
                for index in range(dialog.replacement_combo.count())
            },
            {0, 2, 9},
        )
        self.assertEqual(dialog.replacement_value, 0)
        self.assertNotIn(
            "Tumor",
            " ".join(
                dialog.replacement_combo.itemText(index)
                for index in range(dialog.replacement_combo.count())
            ),
        )

    def test_selected_replacement_value_is_exposed(self):
        dialog = DeleteClassDialog(
            4,
            "Necrosis",
            8,
            {0: "Background", 2: "Stroma", 4: "Necrosis"},
        )

        dialog.replacement_combo.setCurrentIndex(
            dialog.replacement_combo.findData(2)
        )

        self.assertEqual(dialog.replacement_value, 2)
        dialog.replacement_combo.setCurrentIndex(-1)
        self.assertFalse(dialog.delete_button.isEnabled())
        dialog.accept()
        self.assertNotEqual(dialog.result(), QDialog.Accepted)

        dialog.replacement_combo.setCurrentIndex(
            dialog.replacement_combo.findData(2)
        )
        self.assertTrue(dialog.delete_button.isEnabled())
        dialog.delete_button.click()
        self.assertEqual(dialog.result(), QDialog.Accepted)

    def test_background_is_always_available_and_default(self):
        dialog = DeleteClassDialog(3, "Tumor", 1, {3: "Tumor"})

        self.assertEqual(dialog.replacement_combo.count(), 1)
        self.assertEqual(dialog.replacement_value, 0)
        self.assertIn("Background", dialog.replacement_combo.currentText())
        self.assertTrue(dialog.delete_button.isEnabled())

    def test_unused_class_needs_no_meaningful_replacement(self):
        dialog = DeleteClassDialog(5, "Unused", 0, {5: "Unused"})

        self.assertFalse(dialog.replacement_combo.isEnabled())
        self.assertEqual(dialog.replacement_value, 0)
        self.assertIn("unused", dialog.explanation_label.text().lower())
        self.assertTrue(dialog.delete_button.isEnabled())

        dialog.delete_button.click()
        self.assertEqual(dialog.result(), QDialog.Accepted)

    def test_cancel_rejects_dialog(self):
        dialog = DeleteClassDialog(
            2, "Epithelium", 15, {0: "Background", 2: "Epithelium"}
        )

        dialog.cancel_button.click()

        self.assertEqual(dialog.result(), QDialog.Rejected)

    def test_negative_pixel_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            DeleteClassDialog(2, "Epithelium", -1, {0: "Background"})


if __name__ == "__main__":
    unittest.main()
