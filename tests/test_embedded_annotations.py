import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v3 as iio
import numpy as np
import tifffile

from napari_histo_label_editor import _embedded_annotations as embedded
from napari_histo_label_editor._embedded_annotations import (
    PNG_ANNOTATION_CHUNK,
    PNG_SIGNATURE,
    read_embedded_annotations,
    write_image_with_annotations,
)


class EmbeddedAnnotationsTest(unittest.TestCase):
    @staticmethod
    def projection(dtype=np.uint16):
        return np.array(
            [
                [0, 1, 2, 2, 0],
                [1, 1, 3, 2, 0],
                [0, 3, 3, 0, 7],
            ],
            dtype=dtype,
        )

    @staticmethod
    def png_chunks(raw: bytes):
        """Yield complete chunks from one PNG byte string."""
        if not raw.startswith(PNG_SIGNATURE):
            raise AssertionError("test fixture is not a PNG")
        offset = len(PNG_SIGNATURE)
        while offset < len(raw):
            if offset + 8 > len(raw):
                raise AssertionError("test fixture has a truncated chunk header")
            length = struct.unpack(">I", raw[offset : offset + 4])[0]
            stop = offset + 12 + length
            if stop > len(raw):
                raise AssertionError("test fixture has a truncated chunk")
            chunk = raw[offset:stop]
            yield chunk[4:8], chunk
            offset = stop
            if chunk[4:8] == b"IEND":
                break

    def test_legacy_png_and_tiff_have_no_embedded_payload(self):
        projection = self.projection(np.uint8)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for extension in (".png", ".tif", ".tiff"):
                with self.subTest(extension=extension):
                    path = root / f"legacy{extension}"
                    iio.imwrite(path, projection)

                    self.assertIsNone(read_embedded_annotations(path))
                    loaded = iio.imread(path)
                    self.assertEqual(loaded.ndim, 2)
                    self.assertEqual(loaded.dtype, projection.dtype)
                    np.testing.assert_array_equal(loaded, projection)

    def test_payload_roundtrips_inside_same_png_or_tiff(self):
        projection = self.projection()
        payload = bytes(range(256)) * 5 + b"overlap-state\x00\xff"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for extension in (".png", ".tif", ".tiff"):
                with self.subTest(extension=extension):
                    path = root / f"labels{extension}"
                    write_image_with_annotations(path, projection, payload)

                    self.assertEqual(set(root.iterdir()), {path})
                    self.assertEqual(read_embedded_annotations(path), payload)

                    # Generic readers must continue to see an ordinary label
                    # mask rather than a page/channel stack or metadata array.
                    loaded = iio.imread(path)
                    self.assertEqual(loaded.ndim, 2)
                    self.assertEqual(loaded.shape, projection.shape)
                    self.assertEqual(loaded.dtype, projection.dtype)
                    np.testing.assert_array_equal(loaded, projection)

                    path.unlink()

    def test_none_payload_writes_an_ordinary_legacy_image(self):
        projection = self.projection(np.uint8)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.png"

            write_image_with_annotations(path, projection, None)

            self.assertIsNone(read_embedded_annotations(path))
            np.testing.assert_array_equal(iio.imread(path), projection)

    def test_png_payload_uses_small_marker_and_seekable_trailer(self):
        projection = self.projection(np.uint8)
        payload = b"abcdefghijklmnopqrstuvwxyz" * 100
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.png"
            write_image_with_annotations(path, projection, payload)

            annotation_chunks = [
                chunk
                for chunk_type, chunk in self.png_chunks(path.read_bytes())
                if chunk_type == PNG_ANNOTATION_CHUNK
            ]
            self.assertEqual(len(annotation_chunks), 1)
            self.assertLess(len(annotation_chunks[0]), 64)
            self.assertIn(embedded.TRAILER_START_MAGIC, path.read_bytes())
            self.assertEqual(read_embedded_annotations(path), payload)
            np.testing.assert_array_equal(iio.imread(path), projection)

    def test_png_payload_with_corrupted_checksum_is_rejected(self):
        projection = self.projection(np.uint8)
        payload = b"lossless hidden annotations"
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.png"
            write_image_with_annotations(path, projection, payload)
            raw = bytearray(path.read_bytes())
            payload_offset = raw.rindex(embedded.TRAILER_START_MAGIC) + len(
                embedded.TRAILER_START_MAGIC
            )
            raw[payload_offset] ^= 0x01
            path.write_bytes(raw)

            with self.assertRaisesRegex(ValueError, "checksum is invalid"):
                read_embedded_annotations(path)

    def test_truncated_png_payload_is_rejected(self):
        projection = self.projection(np.uint8)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.png"
            write_image_with_annotations(path, projection, b"annotation payload")
            raw = path.read_bytes()
            path.write_bytes(raw[:-7])

            with self.assertRaisesRegex(ValueError, "trailer is missing"):
                read_embedded_annotations(path)

    def test_invalid_png_payload_start_is_rejected_before_reading_payload(self):
        projection = self.projection(np.uint8)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.png"
            write_image_with_annotations(path, projection, b"0123456789")
            raw = bytearray(path.read_bytes())
            start = raw.rindex(embedded.TRAILER_START_MAGIC)
            raw[start] ^= 0x01
            path.write_bytes(raw)

            with self.assertRaisesRegex(ValueError, "payload start is invalid"):
                read_embedded_annotations(path)

    def test_tiff_trailer_corruption_and_truncation_are_rejected(self):
        projection = self.projection(np.uint8)
        payload = b"complete overlap payload"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            corrupt = root / "corrupt.tif"
            write_image_with_annotations(corrupt, projection, payload)
            raw = bytearray(corrupt.read_bytes())
            start = raw.rindex(embedded.TRAILER_START_MAGIC)
            raw[start + len(embedded.TRAILER_START_MAGIC)] ^= 0x01
            corrupt.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "checksum is invalid"):
                read_embedded_annotations(corrupt)

            truncated = root / "truncated.tiff"
            write_image_with_annotations(truncated, projection, payload)
            truncated.write_bytes(truncated.read_bytes()[:-9])
            with self.assertRaisesRegex(ValueError, "trailer is missing"):
                read_embedded_annotations(truncated)

    def test_unrelated_private_tiff_tag_is_not_claimed(self):
        projection = self.projection(np.uint8)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "other-private-tag.tif"
            marker = b"OTHERAPP"
            tifffile.imwrite(
                path,
                projection,
                metadata=None,
                extratags=[
                    (
                        embedded.TIFF_ANNOTATION_TAG,
                        "B",
                        len(marker),
                        marker,
                        False,
                    )
                ],
            )
            self.assertIsNone(read_embedded_annotations(path))
            np.testing.assert_array_equal(iio.imread(path), projection)

    def test_payload_write_rejects_unsupported_extension_without_output(self):
        projection = self.projection(np.uint8)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.bmp"

            with self.assertRaisesRegex(
                ValueError,
                "only in PNG or TIFF",
            ):
                write_image_with_annotations(path, projection, b"payload")

            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
