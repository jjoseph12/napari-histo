import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import imageio.v3 as iio
import numpy as np

from napari_histo_label_editor._io import (
    atomic_save_labels,
    build_image_pyramid,
)


class AtomicSaveLabelsTest(unittest.TestCase):
    def test_casts_and_preserves_shape_and_values(self):
        labels = np.array(
            [[0, 1, 255], [12, 127, 3]],
            dtype=np.uint8,
        )

        with TemporaryDirectory() as tmp:
            destination = Path(tmp) / "labels.tif"
            result = atomic_save_labels(labels, destination, np.int32)
            saved = iio.imread(destination)

        self.assertEqual(result, destination)
        self.assertEqual(saved.shape, labels.shape)
        self.assertEqual(saved.dtype, np.dtype(np.int32))
        np.testing.assert_array_equal(saved, labels)

    def test_atomically_replaces_existing_file_without_leftover_temp(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "labels.tif"
            original = np.full((3, 4), 7, dtype=np.uint16)
            replacement = np.arange(12, dtype=np.uint8).reshape(3, 4)
            iio.imwrite(destination, original)

            atomic_save_labels(replacement, destination, np.uint16)

            np.testing.assert_array_equal(iio.imread(destination), replacement)
            self.assertEqual(list(root.iterdir()), [destination])

    def test_write_failure_keeps_original_and_cleans_temporary_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "labels.tif"
            original = np.full((2, 2), 9, dtype=np.uint8)
            iio.imwrite(destination, original)

            with patch(
                "napari_histo_label_editor._io.iio.imwrite",
                side_effect=OSError("simulated write failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated write failure"):
                    atomic_save_labels(
                        np.zeros((2, 2), dtype=np.uint8),
                        destination,
                        np.uint8,
                    )

            np.testing.assert_array_equal(iio.imread(destination), original)
            self.assertEqual(list(root.iterdir()), [destination])


class BuildImagePyramidTest(unittest.TestCase):
    def test_preserves_level_zero_and_builds_expected_view_shapes(self):
        image = np.arange(17 * 13 * 3, dtype=np.uint16).reshape(17, 13, 3)

        pyramid = build_image_pyramid(image, max_dimension=5)

        self.assertIs(pyramid[0], image)
        self.assertEqual(
            [level.shape for level in pyramid],
            [(17, 13, 3), (9, 7, 3), (5, 4, 3)],
        )
        for level in pyramid[1:]:
            self.assertTrue(np.shares_memory(level, image))

    def test_levels_sample_source_values_at_cumulative_strides(self):
        image = np.arange(16 * 12 * 4, dtype=np.int32).reshape(16, 12, 4)

        pyramid = build_image_pyramid(
            image,
            max_dimension=3,
            downsample=2,
        )

        np.testing.assert_array_equal(pyramid[1], image[::2, ::2, :])
        np.testing.assert_array_equal(pyramid[2], image[::4, ::4, :])
        np.testing.assert_array_equal(pyramid[3], image[::8, ::8, :])

    def test_small_image_returns_only_original_array(self):
        image = np.zeros((32, 40, 3), dtype=np.uint8)

        pyramid = build_image_pyramid(image, max_dimension=40)

        self.assertEqual(len(pyramid), 1)
        self.assertIs(pyramid[0], image)

    def test_rejects_non_rgb_image(self):
        with self.assertRaisesRegex(ValueError, "RGB or RGBA"):
            build_image_pyramid(np.zeros((32, 40), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
