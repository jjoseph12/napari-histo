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

    def test_rejects_lossy_integer_conversion_before_touching_destination(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "labels.tif"
            original = np.full((2, 2), 7, dtype=np.uint8)
            iio.imwrite(destination, original)

            with self.assertRaisesRegex(ValueError, "without data loss"):
                atomic_save_labels(
                    np.array([[0, 300]], dtype=np.uint16),
                    destination,
                    np.uint8,
                )

            np.testing.assert_array_equal(iio.imread(destination), original)
            self.assertEqual(list(root.iterdir()), [destination])


class BuildImagePyramidTest(unittest.TestCase):
    def test_preserves_level_zero_and_builds_expected_filtered_shapes(self):
        image = np.arange(17 * 13 * 3, dtype=np.uint16).reshape(17, 13, 3)

        pyramid = build_image_pyramid(
            image,
            single_scale_limit=8,
            overview_limit=5,
        )

        self.assertIs(pyramid[0], image)
        self.assertEqual(
            [level.shape for level in pyramid],
            [(17, 13, 3), (9, 7, 3), (5, 4, 3)],
        )
        for level in pyramid[1:]:
            self.assertTrue(level.flags.c_contiguous)
            self.assertTrue(level.flags.owndata)
            self.assertFalse(np.shares_memory(level, image))
            self.assertEqual(level.dtype, image.dtype)

    def test_box_filter_prevents_chromatic_stride_aliasing(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[::2, ::2, 1] = 255

        pyramid = build_image_pyramid(
            image,
            single_scale_limit=7,
            overview_limit=4,
            downsample=2,
        )

        self.assertEqual(len(pyramid), 2)
        self.assertTrue(pyramid[1].flags.c_contiguous)
        self.assertTrue(pyramid[1].flags.owndata)
        self.assertTrue(np.all((pyramid[1][..., 1] >= 63)))
        self.assertTrue(np.all((pyramid[1][..., 1] <= 64)))
        self.assertTrue(np.all(pyramid[1][..., (0, 2)] == 0))

    def test_odd_rgba_levels_have_ceil_shapes_and_preserve_dtype(self):
        image = np.full((19, 13, 4), 100.5, dtype=np.float32)

        pyramid = build_image_pyramid(
            image,
            single_scale_limit=10,
            overview_limit=3,
        )

        self.assertEqual(
            [level.shape for level in pyramid],
            [(19, 13, 4), (10, 7, 4), (5, 4, 4), (3, 2, 4)],
        )
        for level in pyramid[1:]:
            self.assertEqual(level.dtype, np.dtype(np.float32))
            self.assertTrue(level.flags.c_contiguous)
            np.testing.assert_allclose(level, 100.5)

    def test_small_image_returns_only_original_array(self):
        image = np.zeros((32, 40, 3), dtype=np.uint8)

        pyramid = build_image_pyramid(
            image,
            single_scale_limit=40,
            overview_limit=20,
        )

        self.assertEqual(len(pyramid), 1)
        self.assertIs(pyramid[0], image)

    def test_default_keeps_current_8001_pixel_image_single_scale(self):
        image = np.zeros((1, 8_001, 3), dtype=np.uint8)

        pyramid = build_image_pyramid(image)

        self.assertEqual(pyramid, [image])
        self.assertIs(pyramid[0], image)

    def test_rejects_non_rgb_image(self):
        with self.assertRaisesRegex(ValueError, "RGB or RGBA"):
            build_image_pyramid(np.zeros((32, 40), dtype=np.uint8))

    def test_validates_trigger_and_overview_limits(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "smaller"):
            build_image_pyramid(
                image,
                single_scale_limit=8,
                overview_limit=8,
            )


if __name__ == "__main__":
    unittest.main()
