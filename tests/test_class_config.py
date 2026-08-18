import csv
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from napari_histo_label_editor._class_config import (
    atomic_write_class_config,
    normalize_color,
    read_class_config,
)
from napari_histo_label_editor._io import file_identity


class NormalizeColorTest(unittest.TestCase):
    def test_normalizes_matplotlib_colors_to_lowercase_hex(self):
        self.assertEqual(normalize_color("RED"), "#ff0000")
        self.assertEqual(normalize_color(" #AbC "), "#aabbcc")
        self.assertEqual(normalize_color((0.0, 0.5, 1.0)), "#0080ff")

    def test_ignores_empty_and_invalid_colors(self):
        self.assertIsNone(normalize_color(None))
        self.assertIsNone(normalize_color(""))
        self.assertIsNone(normalize_color("not a color"))


class ReadClassConfigTest(unittest.TestCase):
    def test_reads_legacy_first_two_columns_and_adds_background(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "classes.csv"
            path.write_text(
                "arbitrary_id,arbitrary_name\n2,Tumor\n7,Stroma\n",
                encoding="utf-8",
            )

            class_map, class_colors = read_class_config(path)

        self.assertEqual(
            class_map,
            {0: "background", 2: "Tumor", 7: "Stroma"},
        )
        self.assertEqual(class_colors, {})

    def test_preserves_explicit_background_name(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "classes.csv"
            path.write_text(
                "value,name\n0,Unlabelled\n1,Tissue\n",
                encoding="utf-8",
            )

            class_map, _ = read_class_config(path)

        self.assertEqual(class_map[0], "Unlabelled")

    def test_reads_case_insensitive_color_column_and_normalizes_values(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "classes.csv"
            path.write_text(
                "value,class_name,notes,CoLoR\n"
                "1,Tumor,primary,red\n"
                "2,Stroma,secondary,#0F8\n"
                "3,Artifact,review,definitely-invalid\n"
                "4,Other,review,\n",
                encoding="utf-8",
            )

            class_map, class_colors = read_class_config(path)

        self.assertEqual(class_map[3], "Artifact")
        self.assertEqual(class_colors, {1: "#ff0000", 2: "#00ff88"})

    def test_reports_non_integer_values_with_the_row_number(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "classes.csv"
            path.write_text("value,name\nnot-an-int,Tumor\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "row 2"):
                read_class_config(path)


class AtomicWriteClassConfigTest(unittest.TestCase):
    def test_rejects_relative_destination(self):
        with self.assertRaisesRegex(ValueError, "absolute path"):
            atomic_write_class_config(Path("classes.csv"), {1: "Tumor"})

    def test_preserves_symlink_and_replaces_its_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "canonical-classes.csv"
            target.write_text("value,name\n1,old\n", encoding="utf-8")
            link = root / "classes.csv"
            link.symlink_to(target.name)

            result = atomic_write_class_config(link, {1: "Tumor"})

            self.assertTrue(link.is_symlink())
            self.assertEqual(result, target.resolve())
            class_map, _ = read_class_config(target)

        self.assertEqual(class_map, {0: "background", 1: "Tumor"})

    def test_rejects_identity_from_file_replaced_before_save(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "classes.csv"
            path.write_text("value,name\n1,loaded\n", encoding="utf-8")
            expected_identity = file_identity(path)
            replacement = root / "replacement.csv"
            replacement.write_text(
                "value,name\n2,external\n",
                encoding="utf-8",
            )
            os.replace(replacement, path)

            with self.assertRaisesRegex(
                RuntimeError,
                "changed since it was loaded",
            ):
                atomic_write_class_config(
                    path,
                    {1: "Editor"},
                    expected_identity=expected_identity,
                )

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "value,name\n2,external\n",
            )
            self.assertEqual(list(root.iterdir()), [path])

    def test_rechecks_identity_immediately_before_replace(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "classes.csv"
            path.write_text("value,name\n1,loaded\n", encoding="utf-8")
            expected_identity = file_identity(path)
            replacement = root / "replacement.csv"
            replacement.write_text(
                "value,name\n2,external-during-save\n",
                encoding="utf-8",
            )
            real_fsync = os.fsync
            real_replace = os.replace

            def replace_destination_after_write(file_descriptor):
                real_fsync(file_descriptor)
                real_replace(replacement, path)

            with patch(
                "napari_histo_label_editor._class_config.os.fsync",
                side_effect=replace_destination_after_write,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "changed since it was loaded",
                ):
                    atomic_write_class_config(
                        path,
                        {1: "Editor"},
                        expected_identity=expected_identity,
                    )

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "value,name\n2,external-during-save\n",
            )
            self.assertEqual(list(root.iterdir()), [path])

    def test_writes_standardized_sorted_csv_and_round_trips(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "classes.csv"

            result = atomic_write_class_config(
                path,
                {8: "Other", 2: "Tumor"},
                {8: "navy", 2: "#ABC", 99: "red"},
            )

            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            class_map, class_colors = read_class_config(path)

        self.assertEqual(result, path.resolve())
        self.assertEqual(
            rows,
            [
                ["value", "class_name", "color"],
                ["0", "background", ""],
                ["2", "Tumor", "#aabbcc"],
                ["8", "Other", "#000080"],
            ],
        )
        self.assertEqual(class_map, {0: "background", 2: "Tumor", 8: "Other"})
        self.assertEqual(class_colors, {2: "#aabbcc", 8: "#000080"})

    def test_invalid_color_is_written_empty_for_palette_fallback(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "classes.csv"
            atomic_write_class_config(
                path,
                {0: "background", 1: "Tumor"},
                {1: "not a real color"},
            )

            _, class_colors = read_class_config(path)

        self.assertEqual(class_colors, {})

    def test_failure_preserves_existing_file_and_cleans_temporary_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "classes.csv"
            original = b"value,name\n1,old\n"
            path.write_bytes(original)

            with patch(
                "napari_histo_label_editor._class_config.os.replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated replace failure"):
                    atomic_write_class_config(path, {1: "new"})

            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(root.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
