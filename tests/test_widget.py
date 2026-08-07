import unittest

import numpy as np
import pandas as pd
from napari.layers import Labels

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


if __name__ == "__main__":
    unittest.main()
