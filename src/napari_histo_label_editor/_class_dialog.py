from __future__ import annotations

from typing import Tuple

from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QColorDialog,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


DEFAULT_CLASS_COLOR = "#16b7c8"


def _normalized_color(value: str) -> str:
    color = QColor(str(value))
    if not color.isValid():
        raise ValueError(f"Invalid class color: {value!r}")
    return color.name().lower()


class ClassEditorDialog(QDialog):
    """Collect the value, name, and display color for a label class."""

    def __init__(
        self,
        value: int,
        name: str,
        color: str,
        *,
        editing: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._editing = bool(editing)
        self._color = _normalized_color(color)

        self.setWindowTitle("Edit class" if editing else "Add class")

        self.value_spin = QSpinBox(self)
        self.value_spin.setRange(1, 2_147_483_647)
        self.value_spin.setValue(int(value))
        self.value_spin.setEnabled(not editing)
        self.value_spin.setToolTip(
            "The numeric value stored in the label image."
        )

        self.name_edit = QLineEdit(self)
        self.name_edit.setText(str(name))
        self.name_edit.setPlaceholderText("Class name")

        self.color_button = QPushButton(self)
        self.color_button.setToolTip("Choose the class display color")
        self.color_button.clicked.connect(self._choose_color)
        self._refresh_color_button()

        form = QFormLayout()
        form.addRow("Class value", self.value_spin)
        form.addRow("Class name", self.name_edit)
        form.addRow("Color", self.color_button)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.button_box)

        self.name_edit.textChanged.connect(self._update_validity)
        self._update_validity()

    @property
    def editing(self) -> bool:
        return self._editing

    @property
    def value(self) -> int:
        return int(self.value_spin.value())

    @value.setter
    def value(self, value: int) -> None:
        self.value_spin.setValue(int(value))

    @property
    def name(self) -> str:
        return self.name_edit.text().strip()

    @name.setter
    def name(self, value: str) -> None:
        self.name_edit.setText(str(value))

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, value: str) -> None:
        self._color = _normalized_color(value)
        self._refresh_color_button()

    def values(self) -> Tuple[int, str, str]:
        """Return a normalized ``(value, name, color)`` tuple."""
        return self.value, self.name, self.color

    def accept(self) -> None:
        if not self.name:
            self._update_validity()
            self.name_edit.setFocus()
            return
        super().accept()

    def _choose_color(self) -> None:
        selected = QColorDialog.getColor(
            QColor(self._color), self, "Choose class color"
        )
        if selected.isValid():
            self.color = selected.name()

    def _update_validity(self, *_args) -> None:
        ok_button = self.button_box.button(QDialogButtonBox.Ok)
        ok_button.setEnabled(bool(self.name))

    def _refresh_color_button(self) -> None:
        color = QColor(self._color)
        foreground = "#000000" if color.lightnessF() > 0.55 else "#ffffff"
        self.color_button.setText(self._color)
        self.color_button.setStyleSheet(
            "QPushButton {"
            f"background-color: {self._color}; color: {foreground};"
            "padding: 5px; border: 1px solid palette(mid);"
            "}"
        )
