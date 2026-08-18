from __future__ import annotations

import warnings
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

from ._class_config import (
    atomic_write_class_config,
    normalize_color,
    read_class_config,
)
from ._class_dialog import ClassEditorDialog
from ._fast_fill import enable_fast_fill
from ._fast_polygon import enable_fast_polygon
from ._fast_rendering import enable_fast_rendering
from ._io import atomic_save_labels, build_image_pyramid

Image.MAX_IMAGE_PIXELS = None

MAX_UNDO_HISTORY = 20
OVERSIZED_TEXTURE_WARNING = (
    r"data shape .* exceeds GL_MAX_TEXTURE_SIZE.*"
)


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
        self.class_colors: Dict[int, str] = {}

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

        layout.addWidget(QLabel("Classes"))

        class_actions = QHBoxLayout()
        self.add_class_btn = QPushButton("+ Add class")
        self.add_class_btn.setEnabled(False)
        self.add_class_btn.clicked.connect(self._add_class)
        class_actions.addWidget(self.add_class_btn)

        self.edit_class_btn = QPushButton("Edit selected")
        self.edit_class_btn.setEnabled(False)
        self.edit_class_btn.clicked.connect(self._edit_selected_class)
        class_actions.addWidget(self.edit_class_btn)
        layout.addLayout(class_actions)
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
            "Usage: select a class, then use napari's native Paint, Fill, "
            "Erase, or Polygon tools. Add classes above; use Edit selected "
            "or right-click a class to rename or recolor it."
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

        self.class_map, self.class_colors = read_class_config(
            self.mapping_path
        )
        self._labels_output_dtype = labels.dtype
        labels = self._compact_labels(labels, self.class_map)
        self._labels_output_dtype = self._promoted_dtype_for_classes(
            self._labels_output_dtype,
            self.class_map,
        )

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

        # This oversized-texture warning is expected and handled immediately
        # below by a balanced display texture; editing data stays full-size.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=OVERSIZED_TEXTURE_WARNING,
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
        self.labels_layer.preserve_labels = False
        self._limit_undo_history(self.labels_layer)
        enable_fast_fill(self.labels_layer)
        enable_fast_polygon(self.labels_layer)
        enable_fast_rendering(self.viewer, self.labels_layer)
        self.viewer.tooltip.visible = True

        self._populate_class_buttons()
        self.add_class_btn.setEnabled(True)
        self.edit_class_btn.setEnabled(True)

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
        background_name = self.class_map.get(0, "background")
        bg_btn = QPushButton(f"0: {background_name}")
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
        bg_btn.setToolTip("Select Background for erasing")

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
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(
                lambda _position, v=value: self._edit_class(v)
            )

            rgba = self._label_rgba(value)
            qcolor = QColor.fromRgbF(rgba[0], rgba[1], rgba[2], 1.0)
            foreground = self._button_text_color(qcolor)
            btn.setStyleSheet(
                f"background-color: {qcolor.name()}; "
                f"color: {foreground}; "
                "font-weight: bold;"
                "padding: 2px;"
                "font-size: 10px;"
            )
            btn.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            btn.setMinimumWidth(0)
            btn.setMaximumWidth(10_000)
            btn.setToolTip(
                f"Click to select {name}. Right-click to rename or recolor."
            )

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

    def _add_class(self) -> None:
        if self.labels_layer is None or self.mapping_path is None:
            self.viewer.status = "Load an image, labels, and class CSV first."
            return

        value = 1
        while value in self.class_map:
            value += 1
        dialog = ClassEditorDialog(
            value,
            "",
            self._class_color_hex(value),
            editing=False,
            parent=self,
        )
        if not self._execute_dialog(dialog):
            return

        value, name, color = dialog.values()
        if value in self.class_map:
            QMessageBox.warning(
                self,
                "Class already exists",
                f"Class value {value} already exists. Edit that class instead.",
            )
            return
        self._apply_class_dialog_values(value, name, color)

    def _edit_selected_class(self) -> None:
        if self.labels_layer is None:
            self.viewer.status = "Load labels first."
            return
        self._edit_class(int(self.labels_layer.selected_label))

    def _edit_class(self, value: int) -> None:
        value = int(value)
        if value == 0:
            self.viewer.status = "Background is reserved and stays transparent."
            return
        if value not in self.class_map:
            self.viewer.status = f"Class {value} is not in the class mapping."
            return

        dialog = ClassEditorDialog(
            value,
            self.class_map[value],
            self._class_color_hex(value),
            editing=True,
            parent=self,
        )
        if self._execute_dialog(dialog):
            self._apply_class_dialog_values(*dialog.values())

    @staticmethod
    def _execute_dialog(dialog: ClassEditorDialog) -> bool:
        """Run a Qt dialog on both Qt 5 and Qt 6 bindings."""
        execute = getattr(dialog, "exec", None)
        if execute is None:
            execute = dialog.exec_
        return bool(execute())

    def _apply_class_dialog_values(
        self,
        value: int,
        name: str,
        color: str,
    ) -> None:
        try:
            self._upsert_class(value, name, color, persist=True)
        except (OSError, TypeError, ValueError, MemoryError) as error:
            self.viewer.status = f"Could not update class: {error}"
            QMessageBox.critical(self, "Could not update class", str(error))

    def _upsert_class(
        self,
        value: int,
        name: str,
        color: str,
        *,
        persist: bool = True,
    ) -> None:
        """Add or update one class and refresh all live layer metadata."""
        value = int(value)
        name = str(name).strip()
        color = normalize_color(color)
        if value <= 0:
            raise ValueError("Class values must be positive; 0 is Background.")
        if not name:
            raise ValueError("Class name cannot be blank.")
        if color is None:
            raise ValueError("Choose a valid class color.")

        new_class_map = dict(self.class_map)
        new_class_map[value] = name
        new_class_colors = dict(self.class_colors)
        new_class_colors[value] = color

        promoted_data = None
        promoted_output_dtype = self._labels_output_dtype
        if self.labels_layer is not None:
            current_data = np.asarray(self.labels_layer.data)
            promoted_dtype = self._promoted_dtype_for_classes(
                current_data.dtype,
                new_class_map,
            )
            if promoted_dtype != current_data.dtype:
                promoted_data = current_data.astype(promoted_dtype)

            output_dtype = (
                current_data.dtype
                if self._labels_output_dtype is None
                else self._labels_output_dtype
            )
            promoted_output_dtype = self._promoted_dtype_for_classes(
                output_dtype,
                new_class_map,
            )

        if persist:
            if self.mapping_path is None:
                raise ValueError("No class mapping CSV is loaded.")
            atomic_write_class_config(
                self.mapping_path,
                new_class_map,
                new_class_colors,
            )

        self.class_map = new_class_map
        self.class_colors = new_class_colors
        self._labels_output_dtype = promoted_output_dtype

        if self.labels_layer is not None:
            if promoted_data is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=OVERSIZED_TEXTURE_WARNING,
                    )
                    self.labels_layer.data = promoted_data
                enable_fast_fill(self.labels_layer)
                enable_fast_polygon(self.labels_layer)
                enable_fast_rendering(self.viewer, self.labels_layer)
            self.labels_layer.features = self._label_features(self.class_map)
            self.labels_layer.colormap = self._multiclass_colormap(
                self.class_map
            )
            self.labels_layer.preserve_labels = False

        self._populate_class_buttons()
        self._select_label(value)
        suffix = " and saved to the class CSV" if persist else ""
        self.viewer.status = f"Class {value}: {name} updated{suffix}."

    @staticmethod
    def _button_text_color(color: QColor) -> str:
        """Choose readable text for a class-colored button."""
        return "#000000" if color.lightnessF() > 0.55 else "#ffffff"

    def _class_color_hex(self, value: int) -> str:
        configured = normalize_color(self.class_colors.get(int(value)))
        if configured is not None:
            return configured
        rgba = self._default_label_rgba(int(value))
        return QColor.fromRgbF(*rgba).name().lower()

    def _label_rgba(self, value: int):
        configured = normalize_color(self.class_colors.get(int(value)))
        if configured is not None:
            return np.array(to_rgba(configured))
        return self._default_label_rgba(value)

    @staticmethod
    def _default_label_rgba(value: int):
        if value <= 0:
            return np.array([0.0, 0.0, 0.0, 0.0])
        if value <= len(CLASS_COLORS):
            return np.array(to_rgba(CLASS_COLORS[value - 1]))

        # fallback for values beyond predefined palette
        rng = np.random.default_rng(value * 1009 + 17)
        rgb = rng.uniform(0.15, 1.0, size=3)
        return np.array([rgb[0], rgb[1], rgb[2], 1.0])

    def _read_class_map(self, path: Path) -> Dict[int, str]:
        """Backward-compatible name-only class mapping reader."""
        return read_class_config(path)[0]

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
    def _promoted_dtype_for_classes(
        current_dtype: np.dtype,
        class_map: Dict[int, str],
    ) -> np.dtype:
        """Keep a dtype when possible, otherwise widen it for new classes."""
        current_dtype = np.dtype(current_dtype)
        if current_dtype == np.dtype(np.bool_):
            minimum, maximum = 0, 1
        elif np.issubdtype(current_dtype, np.integer):
            limits = np.iinfo(current_dtype)
            minimum, maximum = int(limits.min), int(limits.max)
        else:
            raise ValueError(
                f"Label image must have an integer dtype. Got {current_dtype}"
            )

        if class_map:
            minimum = min(minimum, min(int(value) for value in class_map))
            maximum = max(maximum, max(int(value) for value in class_map))

        candidates = (
            (np.uint8, np.uint16, np.uint32, np.uint64)
            if minimum >= 0
            else (np.int8, np.int16, np.int32, np.int64)
        )
        for dtype in candidates:
            dtype = np.dtype(dtype)
            limits = np.iinfo(dtype)
            if limits.min <= minimum and maximum <= limits.max:
                return dtype

        raise ValueError(
            f"Class values from {minimum} to {maximum} exceed supported "
            "integer ranges."
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
