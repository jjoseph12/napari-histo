import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

import imageio.v3 as iio
import numpy as np
import pandas as pd
from napari.components import ViewerModel
from napari.layers import Labels
from qtpy.QtWidgets import QApplication

from napari_histo_label_editor._fast_fill import fast_fill
from napari_histo_label_editor._fast_polygon import fast_paint_polygon
from napari_histo_label_editor._widget import LabelEditorWidget


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
                ["histology", "labels"],
            )
            self.assertEqual(widget.labels_layer.data.dtype, np.dtype(np.uint8))
            self.assertEqual(widget._labels_output_dtype, np.dtype(np.int32))
            self.assertEqual(widget.labels_layer.n_edit_dimensions, 2)
            self.assertTrue(widget.labels_layer.contiguous)
            self.assertEqual(widget.labels_layer._undo_history.maxlen, 20)
            self.assertIs(widget.labels_layer.fill.__func__, fast_fill)
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


if __name__ == "__main__":
    unittest.main()
