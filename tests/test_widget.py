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
from napari_histo_label_editor._class_config import read_class_config
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
    def load_small_project(root, *, labels_dtype=np.uint8):
        image_path = root / "image.png"
        labels_path = root / "labels.tif"
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

    def test_add_and_edit_class_updates_csv_layer_and_color_immediately(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            viewer, widget, _, mapping_path = self.load_small_project(root)

            widget._upsert_class(2, "Stroma", "#123456")

            self.assertEqual(widget.class_map[2], "Stroma")
            self.assertEqual(widget.class_colors[2], "#123456")
            self.assertEqual(widget.labels_layer.selected_label, 2)
            self.assertIs(viewer.layers.selection.active, widget.labels_layer)
            self.assertFalse(widget.labels_layer.preserve_labels)
            np.testing.assert_allclose(
                widget.labels_layer.get_color(2),
                np.array([0x12, 0x34, 0x56, 0xFF]) / 255,
                atol=1 / 255,
            )

            widget.labels_layer.data_setitem(
                (np.array([1]), np.array([1])),
                2,
            )
            self.assertEqual(
                widget.labels_layer._get_tooltip_text((1, 1)),
                "Label: 2 — Stroma",
            )

            widget._upsert_class(2, "Fibrosis", "red")
            class_map, class_colors = read_class_config(mapping_path)
            rows = pd.read_csv(mapping_path)

            self.assertEqual(class_map[2], "Fibrosis")
            self.assertEqual(class_colors[2], "#ff0000")
            self.assertEqual((rows["value"] == 2).sum(), 1)
            self.assertEqual(
                widget.labels_layer._get_tooltip_text((1, 1)),
                "Label: 2 — Fibrosis",
            )
            np.testing.assert_allclose(
                widget.labels_layer.get_color(2),
                np.array([1.0, 0.0, 0.0, 1.0]),
            )

    def test_large_new_class_promotes_edit_and_save_dtypes_losslessly(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, widget, labels_path, _ = self.load_small_project(root)

            self.assertEqual(widget.labels_layer.data.dtype, np.dtype(np.uint8))
            self.assertEqual(widget._labels_output_dtype, np.dtype(np.uint8))

            widget._upsert_class(300, "Review", "#abcdef")

            self.assertEqual(widget.labels_layer.data.dtype, np.dtype(np.uint16))
            self.assertEqual(widget._labels_output_dtype, np.dtype(np.uint16))
            self.assertIs(widget.labels_layer.fill.__func__, fast_fill)
            self.assertIs(
                widget.labels_layer.paint_polygon.__func__,
                fast_paint_polygon,
            )

            widget.labels_layer.data_setitem(
                (np.array([2]), np.array([3])),
                300,
            )
            widget.save_labels()
            self.wait_for_save(widget)
            saved = iio.imread(labels_path)

            self.assertEqual(saved.dtype, np.dtype(np.uint16))
            self.assertEqual(saved[2, 3], 300)


if __name__ == "__main__":
    unittest.main()
