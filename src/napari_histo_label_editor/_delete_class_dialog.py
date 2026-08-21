"""Confirmation dialog for deleting a label class."""

from __future__ import annotations

from collections.abc import Mapping

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


def _button_box_enum(name: str, nested_name: str):
    """Return a QDialogButtonBox enum on Qt 5 and Qt 6."""
    nested = getattr(QDialogButtonBox, nested_name, None)
    if nested is not None:
        return getattr(nested, name)
    return getattr(QDialogButtonBox, name)


class DeleteClassDialog(QDialog):
    """Confirm deletion and, when needed, choose a replacement class.

    Parameters
    ----------
    value, name
        Numeric value and display name of the class being deleted.
    pixel_count
        Number of pixels in the current label image that use ``value``.
    classes
        Existing class names keyed by their numeric values.  The class being
        deleted is excluded from the replacement choices.  Background (value
        zero) is always available and selected by default.
    parent
        Optional parent widget.
    """

    def __init__(
        self,
        value: int,
        name: str,
        pixel_count: int,
        classes: Mapping[int, str],
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.deleted_value = int(value)
        self.deleted_name = str(name).strip()
        self.pixel_count = int(pixel_count)
        if self.pixel_count < 0:
            raise ValueError("Pixel count cannot be negative")

        self.setWindowTitle("Delete class")

        self.class_label = QLabel(
            f"{self.deleted_name} (value {self.deleted_value})", self
        )

        pixel_word = "pixel" if self.pixel_count == 1 else "pixels"
        self.pixel_count_label = QLabel(
            f"{self.pixel_count:,} {pixel_word}", self
        )

        self.replacement_combo = QComboBox(self)
        choices = {
            int(class_value): str(class_name).strip()
            for class_value, class_name in classes.items()
            if int(class_value) != self.deleted_value
        }
        if self.deleted_value != 0:
            choices.setdefault(0, "Background")

        for class_value in sorted(choices):
            class_name = choices[class_value] or f"Class {class_value}"
            self.replacement_combo.addItem(
                f"{class_name} (value {class_value})", class_value
            )

        background_index = self.replacement_combo.findData(0)
        if background_index >= 0:
            self.replacement_combo.setCurrentIndex(background_index)

        if self.pixel_count == 0:
            explanation = (
                "This class is unused, so deleting it will not change any "
                "label pixels. The deletion cannot be undone and clears the "
                "current Undo history. Nothing is written to disk until you "
                "press Save."
            )
            self.replacement_combo.setEnabled(False)
            replacement_row_label = "Replacement (not needed)"
        else:
            explanation = (
                "Choose the class that these pixels should become. This "
                "deletion cannot be undone and clears the current Undo "
                "history. Nothing is written to disk until you press Save."
            )
            replacement_row_label = "Replace pixels with"

        self.explanation_label = QLabel(explanation, self)
        self.explanation_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Class", self.class_label)
        form.addRow("Used by", self.pixel_count_label)
        form.addRow(replacement_row_label, self.replacement_combo)

        self.button_box = QDialogButtonBox(self)
        self.delete_button = self.button_box.addButton(
            "Delete class",
            _button_box_enum("DestructiveRole", "ButtonRole"),
        )
        self.cancel_button = self.button_box.addButton(
            _button_box_enum("Cancel", "StandardButton")
        )
        self.delete_button.clicked.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Make an accidental Return key safer: Cancel, rather than the
        # destructive action, is the default button.
        self.delete_button.setAutoDefault(False)
        self.delete_button.setDefault(False)
        self.cancel_button.setDefault(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.explanation_label)
        layout.addLayout(form)
        layout.addWidget(self.button_box)

        self.replacement_combo.currentIndexChanged.connect(
            self._update_validity
        )
        self._update_validity()

    @property
    def replacement_value(self) -> int:
        """Return the selected replacement's integer label value."""
        data = self.replacement_combo.currentData()
        if data is None:
            # An unused class does not need a replacement.  Returning
            # background keeps the caller's deletion path simple and safe.
            if self.pixel_count == 0:
                return 0
            raise ValueError("No replacement class is available")
        return int(data)

    def accept(self) -> None:
        if self.pixel_count > 0 and self.replacement_combo.currentIndex() < 0:
            self._update_validity()
            self.replacement_combo.setFocus()
            return
        super().accept()

    def _update_validity(self, *_args) -> None:
        has_replacement = self.replacement_combo.currentIndex() >= 0
        self.delete_button.setEnabled(self.pixel_count == 0 or has_replacement)
