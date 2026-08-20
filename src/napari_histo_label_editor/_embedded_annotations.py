"""Embed lossless overlap annotations inside an ordinary label image.

The first image in the file remains a conventional two-dimensional
categorical mask.  Readers that do not know about this plugin therefore see
the usual topmost-label projection, while the plugin can recover the complete
overlap model from private, standards-compliant metadata in the same file.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path
from typing import Union

import imageio.v3 as iio
import numpy as np
import tifffile


PathLike = Union[str, "Path"]

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_ANNOTATION_CHUNK = b"npAR"
PNG_PART_MAGIC = b"NHLAPNG1"
PNG_TRAILER_MARKER = b"NHLAPNG2"
PNG_PART_HEADER = struct.Struct(">8sII")
PNG_MAX_PART_BYTES = 64 * 1024 * 1024
PNG_MAX_PARTS = 65_536
PNG_MAX_CHUNK_BYTES = 0x7FFFFFFF
TIFF_ANNOTATION_TAG = 65000
TIFF_TRAILER_MARKER = b"NHLATIF2"
TRAILER_START_MAGIC = b"NHLADATA"
TRAILER_FOOTER_MAGIC = b"NHLAFTR1"
TRAILER_FOOTER = struct.Struct(">8sQ32s")

__all__ = [
    "PNG_ANNOTATION_CHUNK",
    "TIFF_ANNOTATION_TAG",
    "read_embedded_annotations",
    "write_image_with_annotations",
]


def _extension(path: PathLike) -> str:
    return Path(path).suffix.lower()


def write_image_with_annotations(
    path: PathLike,
    image: np.ndarray,
    payload: bytes | None,
) -> None:
    """Write one normal label image and an optional private annotation payload.

    PNG readers ignore the private ancillary chunks, and TIFF readers ignore
    the private tag.  In both cases the ordinary image remains a 2-D mask.
    """
    destination = Path(path)
    extension = _extension(destination)
    if payload is None:
        iio.imwrite(destination, image)
        return
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("Embedded annotation payload must be bytes-like")
    payload = bytes(payload)
    if extension == ".png":
        iio.imwrite(destination, image)
        _append_png_payload(destination, payload)
        return
    if extension in {".tif", ".tiff"}:
        # Keep only a tiny marker in the TIFF directory. Generic TIFF readers
        # eagerly materialize large private tags, which would make opening a
        # normal projection needlessly allocate the complete overlap bundle.
        # The lossless payload lives in an unreferenced trailer and is found
        # in O(1) time from the fixed footer at EOF.
        tifffile.imwrite(
            destination,
            image,
            bigtiff=(np.asarray(image).nbytes >= 0xF0000000),
            metadata=None,
            extratags=[
                (
                    TIFF_ANNOTATION_TAG,
                    "B",
                    len(TIFF_TRAILER_MARKER),
                    TIFF_TRAILER_MARKER,
                    False,
                )
            ],
        )
        _append_annotation_trailer(destination, payload)
        return
    raise ValueError(
        "Lossless overlapping annotations can be embedded only in PNG or "
        f"TIFF label images. Got: {destination.suffix or '<no extension>'}"
    )


def read_embedded_annotations(path: PathLike) -> bytes | None:
    """Return the embedded overlap payload, or ``None`` for a legacy image."""
    source = Path(path)
    extension = _extension(source)
    if extension == ".png":
        return _read_png_payload(source)
    if extension in {".tif", ".tiff"}:
        with tifffile.TiffFile(source) as tif:
            if not tif.pages:
                raise ValueError(f"TIFF label image has no pages: {source}")
            tag = tif.pages[0].tags.get(TIFF_ANNOTATION_TAG)
            if tag is None:
                return None
            value = tag.value
        if isinstance(value, bytes):
            marker = value
        if isinstance(value, np.ndarray):
            marker = value.astype(np.uint8, copy=False).tobytes()
        elif not isinstance(value, bytes):
            try:
                marker = bytes(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Embedded TIFF annotation marker is invalid: {source}"
                ) from error
        if marker != TIFF_TRAILER_MARKER:
            # Tag 65000 is private and another TIFF producer may legitimately
            # use it. Only our exact marker claims the trailer contract.
            return None
        return _read_annotation_trailer(source, "TIFF")
    return None


def _append_annotation_trailer(path: Path, payload: bytes) -> None:
    checksum = hashlib.sha256(payload).digest()
    with path.open("ab") as stream:
        stream.write(TRAILER_START_MAGIC)
        stream.write(payload)
        stream.write(
            TRAILER_FOOTER.pack(
                TRAILER_FOOTER_MAGIC,
                len(payload),
                checksum,
            )
        )


def _read_annotation_trailer(path: Path, format_name: str) -> bytes:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        file_size = stream.tell()
        minimum_size = len(TRAILER_START_MAGIC) + TRAILER_FOOTER.size
        if file_size < minimum_size:
            raise ValueError(
                f"Embedded {format_name} annotation trailer is missing: {path}"
            )
        stream.seek(-TRAILER_FOOTER.size, 2)
        footer = stream.read(TRAILER_FOOTER.size)
        magic, payload_length, expected_checksum = (
            TRAILER_FOOTER.unpack(footer)
        )
        if magic != TRAILER_FOOTER_MAGIC:
            raise ValueError(
                f"Embedded {format_name} annotation trailer is missing: {path}"
            )
        available = file_size - TRAILER_FOOTER.size
        if payload_length > available - len(TRAILER_START_MAGIC):
            raise ValueError(
                f"Embedded {format_name} annotation payload is truncated: {path}"
            )
        start_offset = available - payload_length - len(TRAILER_START_MAGIC)
        stream.seek(start_offset)
        if stream.read(len(TRAILER_START_MAGIC)) != TRAILER_START_MAGIC:
            raise ValueError(
                f"Embedded {format_name} annotation payload start is invalid: "
                f"{path}"
            )
        payload = stream.read(payload_length)
    if hashlib.sha256(payload).digest() != expected_checksum:
        raise ValueError(
            f"Embedded {format_name} annotation payload checksum is invalid: "
            f"{path}"
        )
    return payload


def _append_png_payload(path: Path, payload: bytes) -> None:
    with path.open("r+b") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        if size < len(PNG_SIGNATURE) + 12:
            raise ValueError(f"PNG output is truncated: {path}")
        stream.seek(0)
        if stream.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            raise ValueError(f"PNG output has an invalid signature: {path}")
        stream.seek(-12, 2)
        iend = stream.read(12)
        if iend != b"\x00\x00\x00\x00IEND\xaeB`\x82":
            raise ValueError(f"PNG output does not end with IEND: {path}")

        stream.seek(-12, 2)
        stream.write(struct.pack(">I", len(PNG_TRAILER_MARKER)))
        stream.write(PNG_ANNOTATION_CHUNK)
        stream.write(PNG_TRAILER_MARKER)
        checksum = zlib.crc32(PNG_ANNOTATION_CHUNK)
        checksum = zlib.crc32(PNG_TRAILER_MARKER, checksum) & 0xFFFFFFFF
        stream.write(struct.pack(">I", checksum))
        stream.write(iend)
        stream.truncate()
    _append_annotation_trailer(path, payload)


def _read_png_payload(path: Path) -> bytes | None:
    payload = bytearray()
    part_count = 0
    expected_total = None
    trailer_marked = False
    saw_iend = False
    with path.open("rb") as stream:
        if stream.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            raise ValueError(f"PNG label image has an invalid signature: {path}")
        while True:
            length_bytes = stream.read(4)
            if not length_bytes:
                break
            if len(length_bytes) != 4:
                raise ValueError(f"PNG chunk header is truncated: {path}")
            length = struct.unpack(">I", length_bytes)[0]
            if length > PNG_MAX_CHUNK_BYTES:
                raise ValueError(f"PNG chunk is too large: {path}")
            chunk_type = stream.read(4)
            if len(chunk_type) != 4:
                raise ValueError(f"PNG chunk is truncated: {path}")
            actual_crc = zlib.crc32(chunk_type)
            if chunk_type == PNG_ANNOTATION_CHUNK:
                if length == len(PNG_TRAILER_MARKER):
                    marker = stream.read(length)
                    if len(marker) != length:
                        raise ValueError(f"PNG chunk is truncated: {path}")
                    actual_crc = zlib.crc32(marker, actual_crc)
                    if marker != PNG_TRAILER_MARKER:
                        # It may be a legacy multipart header of the same
                        # length only by coincidence; normal legacy parts are
                        # longer than their 16-byte header.
                        raise ValueError(
                            f"Embedded PNG annotation marker is invalid: {path}"
                        )
                    trailer_marked = True
                    remaining = 0
                else:
                    remaining = length
                if remaining and length < PNG_PART_HEADER.size:
                    raise ValueError(
                        f"Embedded PNG annotation chunk is truncated: {path}"
                    )
                if remaining and length > PNG_PART_HEADER.size + PNG_MAX_PART_BYTES:
                    raise ValueError(
                        f"Embedded PNG annotation part is too large: {path}"
                    )
                if remaining:
                    header = stream.read(PNG_PART_HEADER.size)
                    if len(header) != PNG_PART_HEADER.size:
                        raise ValueError(f"PNG chunk is truncated: {path}")
                    actual_crc = zlib.crc32(header, actual_crc)
                    magic, part_index, total_parts = PNG_PART_HEADER.unpack(header)
                    if magic != PNG_PART_MAGIC or total_parts < 1:
                        raise ValueError(
                            f"Embedded PNG annotation header is invalid: {path}"
                        )
                    if total_parts > PNG_MAX_PARTS:
                        raise ValueError(
                            f"Embedded PNG annotation has too many parts: {path}"
                        )
                    if expected_total is None:
                        expected_total = total_parts
                    elif expected_total != total_parts:
                        raise ValueError(
                            f"Embedded PNG annotation parts disagree: {path}"
                        )
                    if part_index != part_count:
                        raise ValueError(
                            "Embedded PNG annotation payload is incomplete or "
                            f"out of order: {path}"
                        )
                    remaining = length - PNG_PART_HEADER.size
                    while remaining:
                        block = stream.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError(f"PNG chunk is truncated: {path}")
                        payload.extend(block)
                        actual_crc = zlib.crc32(block, actual_crc)
                        remaining -= len(block)
                    part_count += 1
            else:
                remaining = length
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError(f"PNG chunk is truncated: {path}")
                    actual_crc = zlib.crc32(block, actual_crc)
                    remaining -= len(block)
            crc_bytes = stream.read(4)
            if len(crc_bytes) != 4:
                raise ValueError(f"PNG chunk is truncated: {path}")
            expected_crc = struct.unpack(">I", crc_bytes)[0]
            if (actual_crc & 0xFFFFFFFF) != expected_crc:
                raise ValueError(f"PNG chunk checksum is invalid: {path}")
            if chunk_type == b"IEND":
                if length != 0:
                    raise ValueError(f"PNG IEND chunk must be empty: {path}")
                saw_iend = True
                break

    if not saw_iend:
        raise ValueError(f"PNG label image is missing IEND: {path}")
    if trailer_marked:
        if expected_total is not None:
            raise ValueError(
                f"PNG contains conflicting embedded annotation formats: {path}"
            )
        return _read_annotation_trailer(path, "PNG")
    if expected_total is None:
        return None
    if part_count != expected_total:
        raise ValueError(f"Embedded PNG annotation payload is incomplete: {path}")
    return bytes(payload)
