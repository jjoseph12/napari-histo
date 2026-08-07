from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QMessageBox, QGridLayout, 
    QSizePolicy, QScrollArea
)
from qtpy.QtGui import QColor
from qtpy.QtCore import Qt 
from napari.viewer import Viewer
from napari.utils.colormaps import DirectLabelColormap
from matplotlib.colors import to_rgba

Image.MAX_IMAGE_PIXELS = None


CLASS_COLORS = [
    '#16b7c8',
    '#f9faaf',
    '#b8b4d4',
    '#b6b620',
    '#89cdc2',
    '#1e73ae',
    '#2a9b2b',
    '#f97c0d',
    '#dd75bd',
    '#cf2526',
    '#9063b8',
    '#89534a',
    '#7c7c7c',
    '#c5e4c0',
    '#7cabcd',
    '#f5c7df',
    '#f47c6f',
    '#d5d5d5',
    '#b87db9',
    '#aed866',
    '#f7b160'
]

class LabelEditorWidget(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.image_path: Optional[Path] = None
        self.labels_path: Optional[Path] = None
        self.mapping_path: Optional[Path] = None

        self.class_button_layout = None

        self.class_map: Dict[int, str] = {}
        self.undo_stack = []
        self.max_undo = 20

        self.labels_layer = None
        self.undo_stack = []
        self.max_undo = 20

        self._build_ui()
        self._bind_hotkeys()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.image_line = QLineEdit()
        self.label_line = QLineEdit()
        self.mapping_line = QLineEdit()

        layout.addWidget(QLabel("Histology image"))
        layout.addLayout(self._file_row(self.image_line, self._choose_image))

        layout.addWidget(QLabel("Label image"))
        layout.addLayout(self._file_row(self.label_line, self._choose_labels))

        layout.addWidget(QLabel("Class mapping CSV"))
        layout.addLayout(self._file_row(self.mapping_line, self._choose_mapping))

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_data)
        layout.addWidget(load_btn)

        layout.addWidget(QLabel("Class shortcuts"))
        self.class_button_container = QWidget()
        self.class_button_container.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Maximum,
        )

        self.class_button_layout = QGridLayout(self.class_button_container)
        self.class_button_layout.setContentsMargins(0, 0, 0, 0)
        self.class_button_layout.setSpacing(2)

        self.class_button_scroll = QScrollArea()
        self.class_button_scroll.setWidgetResizable(True)
        self.class_button_scroll.setWidget(self.class_button_container)
        self.class_button_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.class_button_scroll.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Expanding,
        )
        self.class_button_scroll.setMinimumWidth(280)
        self.class_button_scroll.setMinimumHeight(320)
        self.class_button_scroll.setMaximumHeight(600)

        layout.addWidget(self.class_button_scroll)

        save_btn = QPushButton("Save [s]")
        save_btn.clicked.connect(self.save_labels)
        layout.addWidget(save_btn)

        undo_btn = QPushButton("Undo [u]")
        undo_btn.clicked.connect(self.undo)
        layout.addWidget(undo_btn)

        layout.addWidget(QLabel(
            "Usage: select a class channel, then use napari's native Paint, Fill, "
            "Erase, or Polygon tools. Each class is a distinct label in the same layer."
        ))

        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(260)
        self.resize(320, self.height())

    def _file_row(self, line_edit, callback):
        row = QHBoxLayout()
        browse = QPushButton("Browse")
        browse.clicked.connect(callback)
        row.addWidget(line_edit)
        row.addWidget(browse)
        return row

    def _choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose histology image")
        if path:
            self.image_line.setText(path)

    def _choose_labels(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose label image")
        if path:
            self.label_line.setText(path)

    def _choose_mapping(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose class mapping CSV")
        if path:
            self.mapping_line.setText(path)

    def _bind_hotkeys(self):
        @self.viewer.bind_key("s", overwrite=True)
        def _save(viewer):
            self.save_labels()

        @self.viewer.bind_key("u", overwrite=True)
        def _undo(viewer):
            self.undo()

    def load_data(self):
        self.image_path = Path(self.image_line.text())
        self.labels_path = Path(self.label_line.text())
        self.mapping_path = Path(self.mapping_line.text())

        image = iio.imread(self.image_path)
        labels = iio.imread(self.labels_path)

        if labels.ndim != 2:
            raise ValueError(f"Label image must be 2D. Got {labels.shape}")

        if image.shape[:2] != labels.shape:
            raise ValueError(
                f"Image and labels differ: {image.shape[:2]} vs {labels.shape}"
            )

        labels = labels.astype(np.int32, copy=False)
        self.class_map = self._read_class_map(self.mapping_path)

        self.viewer.layers.clear()
        self.undo_stack.clear()

        self.viewer.add_image(
            image,
            name="histology",
            rgb=image.ndim == 3 and image.shape[-1] in (3, 4),
        )

        self.viewer.layers.clear()
        self.undo_stack.clear()

        self.viewer.add_image(
            image,
            name="histology",
            rgb=image.ndim == 3 and image.shape[-1] in (3, 4),
        )

        self.labels_layer = self.viewer.add_labels(
            labels,
            name="labels",
            opacity=0.45,
            features=self._label_features(self.class_map),
        )

        self.labels_layer.colormap = self._multiclass_colormap(self.class_map)
        self.labels_layer.contour = 0
        self.labels_layer.selected_label = 1
        self.labels_layer.refresh()
        self.viewer.tooltip.visible = True

        self.labels_layer.mouse_drag_callbacks.append(self._snapshot_on_mouse_press)

        self._populate_class_buttons()

        self.viewer.status = (
            "Loaded multiclass label layer. Hover over an area to see its label."
        )

    def _populate_class_buttons(self):
        # Clear old buttons
        while self.class_button_layout.count():
            item = self.class_button_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        row = 0
        col = 0
        n_cols = max(1, min(3, self.width() // 170))

        # ------------------------------------------------------------------
        # Background button
        # ------------------------------------------------------------------
        bg_btn = QPushButton("0: Background")
        bg_btn.clicked.connect(lambda checked=False: self._select_label(0))
        bg_btn.setStyleSheet(
            "background-color: #d9d9d9;"
            "color: black;"
            "font-weight: bold;"
            "padding: 2px;"
            "font-size: 10px;"
        )
        bg_btn.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        bg_btn.setMinimumWidth(0)
        bg_btn.setMaximumWidth(10_000)
        bg_btn.setToolTip(bg_btn.text())

        self.class_button_layout.addWidget(bg_btn, row, col)

        col += 1
        if col >= n_cols:
            col = 0
            row += 1

        # ------------------------------------------------------------------
        # Class buttons
        # ------------------------------------------------------------------

        for value, name in sorted(self.class_map.items()):
            if value == 0:
                continue

            btn = QPushButton(f"{value}: {name}")
            btn.clicked.connect(lambda checked=False, v=value: self._select_label(v))

            rgba = self._label_rgba(value)
            qcolor = QColor.fromRgbF(rgba[0], rgba[1], rgba[2], 1.0)
            btn.setStyleSheet(
                f"background-color: {qcolor.name()}; "
                "color: black; "
                "font-weight: bold;"
                "padding: 2px;"
                "font-size: 10px;"
            )
            btn.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            btn.setMinimumWidth(0)
            btn.setMaximumWidth(10_000)
            btn.setToolTip(btn.text())

            self.class_button_layout.addWidget(btn, row, col)

            col += 1
            if col >= n_cols:
                col = 0
                row += 1


    def _select_label(self, value: int):
        if self.labels_layer is None:
            return

        self.labels_layer.selected_label = int(value)
        self.viewer.layers.selection.active = self.labels_layer
        self.viewer.status = f"Selected label {value}: {self.class_map.get(value, value)}"


    def _label_rgba(self, value: int):
        if value <= len(CLASS_COLORS):
            return np.array(to_rgba(CLASS_COLORS[value - 1]))

        # fallback for values beyond predefined palette
        rng = np.random.default_rng(value * 1009 + 17)
        rgb = rng.uniform(0.15, 1.0, size=3)
        return np.array([rgb[0], rgb[1], rgb[2], 1.0])

    def _read_class_map(self, path: Path) -> Dict[int, str]:
        df = pd.read_csv(path)
        value_col = df.columns[0]
        name_col = df.columns[1]

        mapping = {
            int(row[value_col]): str(row[name_col])
            for _, row in df.iterrows()
        }

        mapping.setdefault(0, "background")
        return mapping

    @staticmethod
    def _label_features(class_map: Dict[int, str]) -> pd.DataFrame:
        """Build label metadata used by napari's status bar and tooltips."""
        values = sorted(class_map)
        return pd.DataFrame(
            {
                "index": values,
                "Label": [f"{value} — {class_map[value]}" for value in values],
            }
        )

    def _multiclass_colormap(self, class_map):
        color_dict = {
            None: np.array([0, 0, 0, 0]),
            0: np.array([0, 0, 0, 0]),
        }

        for value in sorted(class_map):
            if value == 0:
                continue
            color_dict[int(value)] = self._label_rgba(int(value))

        return DirectLabelColormap(color_dict=color_dict)

    def _snapshot(self):
        if self.labels_layer is None:
            print("snapshot skipped: no labels_layer")
            return

        #print("snapshot taken")
        self.undo_stack.append(np.asarray(self.labels_layer.data).copy())

        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def _snapshot_on_mouse_press(self, layer, event):
        if event.type == "mouse_press":
            self._snapshot()

    def undo(self):
        #print(f"undo stack size: {len(self.undo_stack)}")

        if self.labels_layer is None or not self.undo_stack:
            self.viewer.status = "Nothing to undo."
            return

        self.labels_layer.data = self.undo_stack.pop()
        self.labels_layer.refresh()
        self.viewer.status = "Undo complete."

    def save_labels(self):
        if self.labels_path is None or self.labels_layer is None:
            self.viewer.status = "No labels loaded."
            return

        try:
            labels = np.asarray(self.labels_layer.data).astype(np.int32)
            iio.imwrite(self.labels_path, labels)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return

        self.viewer.status = f"Saved labels to {self.labels_path}"
