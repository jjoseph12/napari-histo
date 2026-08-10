from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image
from napari.qt.threading import create_worker
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

from ._fast_fill import enable_fast_fill
from ._fast_polygon import enable_fast_polygon
from ._fast_rendering import enable_fast_rendering
from ._io import atomic_save_labels, build_image_pyramid

Image.MAX_IMAGE_PIXELS = None

MAX_UNDO_HISTORY = 20


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
        self._labels_output_dtype: Optional[np.dtype] = None
        self._save_worker = None
        self._labels_was_editable = True

        self.class_button_layout = None

        self.class_map: Dict[int, str] = {}

        self.labels_layer = None

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

        self.save_btn = QPushButton("Save [s]")
        self.save_btn.clicked.connect(self.save_labels)
        layout.addWidget(self.save_btn)

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

        self.class_map = self._read_class_map(self.mapping_path)
        self._labels_output_dtype = labels.dtype
        labels = self._compact_labels(labels, self.class_map)

        self.viewer.layers.clear()

        is_rgb = image.ndim == 3 and image.shape[-1] in (3, 4)
        image_pyramid = build_image_pyramid(image) if is_rgb else [image]
        image_data = image_pyramid if len(image_pyramid) > 1 else image
        self.viewer.add_image(
            image_data,
            name="histology",
            rgb=is_rgb,
            multiscale=len(image_pyramid) > 1,
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
        self.labels_layer.n_edit_dimensions = 2
        self.labels_layer.contiguous = True
        self._limit_undo_history(self.labels_layer)
        enable_fast_fill(self.labels_layer)
        enable_fast_polygon(self.labels_layer)
        enable_fast_rendering(self.viewer, self.labels_layer)
        self.viewer.tooltip.visible = True

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

    @staticmethod
    def _compact_labels(
        labels: np.ndarray,
        class_map: Dict[int, str],
    ) -> np.ndarray:
        """Use the smallest integer dtype that can represent all labels.

        The editor previously forced every mask to int32. For a large semantic
        mask with only a few classes, that needlessly quadruples memory use.
        """
        labels = np.asarray(labels)
        is_integer = np.issubdtype(labels.dtype, np.integer)
        if not (is_integer or labels.dtype == np.bool_):
            raise ValueError(
                f"Label image must have an integer dtype. Got {labels.dtype}"
            )

        if labels.size:
            minimum = int(labels.min())
            maximum = int(labels.max())
        else:
            minimum = maximum = 0

        if class_map:
            minimum = min(minimum, min(class_map))
            maximum = max(maximum, max(class_map))

        candidates = (
            (np.uint8, np.uint16, np.uint32, np.uint64)
            if minimum >= 0
            else (np.int8, np.int16, np.int32, np.int64)
        )
        for dtype in candidates:
            limits = np.iinfo(dtype)
            if limits.min <= minimum and maximum <= limits.max:
                return labels.astype(dtype, copy=False)

        raise ValueError(
            f"Label values from {minimum} to {maximum} exceed supported integer ranges"
        )

    @staticmethod
    def _undo_labels_layer(layer) -> bool:
        """Undo one edit using napari's sparse, changed-pixel history."""
        history = getattr(layer, "_undo_history", None)
        undo = getattr(layer, "undo", None)
        if undo is None or (history is not None and not history):
            return False
        undo()
        return True

    @staticmethod
    def _limit_undo_history(layer, limit: int = MAX_UNDO_HISTORY) -> None:
        """Bound napari's sparse edit history for predictable memory use."""
        reset_history = getattr(layer, "_reset_history", None)
        if reset_history is None:
            return
        layer._history_limit = int(limit)
        reset_history()

    def undo(self):
        if self.labels_layer is None or not self._undo_labels_layer(
            self.labels_layer
        ):
            self.viewer.status = "Nothing to undo."
            return

        self.viewer.status = "Undo complete."

    def save_labels(self):
        if self.labels_path is None or self.labels_layer is None:
            self.viewer.status = "No labels loaded."
            return

        if self._save_worker is not None and self._save_worker.is_running:
            self.viewer.status = "A label save is already running."
            return

        labels = np.asarray(self.labels_layer.data)
        output_dtype = (
            labels.dtype
            if self._labels_output_dtype is None
            else self._labels_output_dtype
        )
        self._labels_was_editable = self.labels_layer.editable
        self.labels_layer.editable = False
        self.save_btn.setEnabled(False)
        self.viewer.status = f"Saving labels to {self.labels_path}…"

        worker = create_worker(
            atomic_save_labels,
            labels,
            self.labels_path,
            output_dtype,
            _start_thread=False,
            _ignore_errors=True,
        )
        worker.returned.connect(self._on_save_complete)
        worker.errored.connect(self._on_save_error)
        worker.finished.connect(self._on_save_finished)
        self._save_worker = worker
        worker.start()

    def _on_save_complete(self, saved_path: Path) -> None:
        self.viewer.status = f"Saved labels to {saved_path}"

    def _on_save_error(self, error: Exception) -> None:
        self.viewer.status = f"Save failed: {error}"
        QMessageBox.critical(self, "Save failed", str(error))

    def _on_save_finished(self) -> None:
        if self.labels_layer is not None:
            self.labels_layer.editable = self._labels_was_editable
        self.save_btn.setEnabled(True)
        self._save_worker = None
