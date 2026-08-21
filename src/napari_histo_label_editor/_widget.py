from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Dict, Optional

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image
from napari.qt.threading import create_worker
from napari.layers.labels._labels_constants import Mode
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QMessageBox, QGridLayout, 
    QSizePolicy, QScrollArea, QSlider
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
from ._delete_class_dialog import DeleteClassDialog
from ._embedded_annotations import read_embedded_annotations
from ._fast_fill import enable_overlap_fill
from ._fast_polygon import enable_fast_polygon
from ._fast_rendering import (
    enable_fast_rendering,
    enable_fast_texture_updates,
)
from ._io import (
    FileIdentity,
    SaveResult,
    atomic_save_labels,
    build_image_pyramid,
    file_identity,
)
from ._overlap_editor import OverlapEditorController
from ._overlap_store import OverlapStore

Image.MAX_IMAGE_PIXELS = None

MAX_UNDO_HISTORY = 20
CLASS_SCAN_MASK_BYTES = 8 * 1024 * 1024


def _tracked_overlap_data_setitem(layer, indices, value, refresh=True):
    return layer._napari_histo_edit_tracker.data_setitem(
        indices,
        value,
        refresh,
    )


@contextmanager
def _tracked_overlap_block_history(layer):
    with layer._napari_histo_edit_tracker.block_history():
        yield


def _tracked_overlap_undo(layer):
    return layer._napari_histo_edit_tracker.undo()


def _tracked_overlap_redo(layer):
    return layer._napari_histo_edit_tracker.redo()


class _OverlapEditTracker:
    """Pair napari's sparse binary history with sparse visible-top history."""

    def __init__(self, widget, layer) -> None:
        self.widget = widget
        self.layer = layer
        self.native_data_setitem = layer.data_setitem
        self.native_block_history = layer.block_history
        self.native_undo = layer.undo
        self.native_redo = layer.redo
        limit = int(getattr(layer, "_history_limit", MAX_UNDO_HISTORY))
        self.undo_items = deque(maxlen=limit)
        self.redo_items = deque(maxlen=limit)
        self.staged = []
        self.block_depth = 0

    def install(self) -> None:
        layer = self.layer
        layer._napari_histo_edit_tracker = self
        layer.data_setitem = MethodType(_tracked_overlap_data_setitem, layer)
        layer.block_history = MethodType(
            _tracked_overlap_block_history,
            layer,
        )
        layer.undo = MethodType(_tracked_overlap_undo, layer)
        layer.redo = MethodType(_tracked_overlap_redo, layer)

    def clear(self) -> None:
        self.undo_items.clear()
        self.redo_items.clear()
        self.staged.clear()

    def _native_history_marker(self):
        history = (
            self.layer._staged_history
            if self.layer._block_history
            else self.layer._undo_history
        )
        return None if not history else id(history[-1])

    def _add_empty_native_history_atom(self) -> None:
        empty_indices = tuple(
            np.empty(0, dtype=np.intp) for _ in range(self.layer.ndim)
        )
        empty_values = np.empty(0, dtype=np.asarray(self.layer.data).dtype)
        self.layer._save_history(
            (empty_indices, empty_values, empty_values.copy())
        )

    def data_setitem(self, indices, value, refresh=True):
        if self.widget._save_worker is not None:
            # Mode.PAN_ZOOM is the normal UX lock, but callers can change
            # mode programmatically while the save worker owns the store.
            # Reject before touching napari data/history so its native queue
            # cannot diverge from the packed membership/top history.
            self.widget.viewer.status = (
                "Wait for the current label save to finish before editing."
            )
            return None
        controller = self.widget.overlap_editor
        normalized = controller.normalize_indices(indices)
        rows, columns = normalized
        binary_before = np.array(
            self.layer.data[rows, columns],
            copy=True,
        )
        top_before = np.array(
            controller.composite[rows, columns],
            copy=True,
        )
        history_before = self._native_history_marker()

        result = self.native_data_setitem(indices, value, refresh)
        history_after = self._native_history_marker()
        native_recorded = history_after != history_before

        if (
            self.widget._syncing_overlap_layer
            or self.widget._save_worker is not None
            or not self.widget._labels_layer_is_active()
            or controller.active_class is None
        ):
            return result

        membership_changed = controller.process_indices(normalized)
        top_changed = 0
        value_array = np.asarray(value)
        if value_array.ndim == 0 and int(value_array) == 1:
            top_changed, _ = controller.raise_active_indices(normalized)

        binary_after = np.array(
            self.layer.data[rows, columns],
            copy=True,
        )
        top_after = np.array(
            controller.composite[rows, columns],
            copy=True,
        )
        recorded = (binary_before != binary_after) | (top_before != top_after)
        if np.any(recorded):
            atom = (
                (rows[recorded].copy(), columns[recorded].copy()),
                top_before[recorded].copy(),
                top_after[recorded].copy(),
            )
            if not native_recorded:
                # napari drops binary 1 -> 1 edits. Keep an empty native atom
                # so its queue stays aligned with this top-only history item.
                self._add_empty_native_history_atom()
            self._record(atom)

        if membership_changed or top_changed:
            bounds = controller._bounds_for_coordinates(rows, columns)
            self.widget._refresh_composite_bounds(bounds)
        return result

    def _record(self, atom) -> None:
        self.redo_items.clear()
        if self.layer._block_history:
            self.staged.append(atom)
        else:
            self.undo_items.append([atom])

    @contextmanager
    def block_history(self):
        self.block_depth += 1
        completed = False
        try:
            with self.native_block_history():
                yield
            completed = True
        finally:
            self.block_depth -= 1
            if self.block_depth == 0:
                if completed and self.staged:
                    self.undo_items.append(self.staged)
                self.staged = []

    def undo(self):
        if self.widget._save_worker is not None:
            self.widget.viewer.status = (
                "Wait for the current label save to finish before undoing."
            )
            return None
        if not self.layer._undo_history:
            return None
        history = self.undo_items.pop() if self.undo_items else None
        result = self.native_undo()
        if history is not None:
            self.widget._restore_overlap_history(history, undoing=True)
            self.redo_items.append(history)
        return result

    def redo(self):
        if self.widget._save_worker is not None:
            self.widget.viewer.status = (
                "Wait for the current label save to finish before redoing."
            )
            return None
        if not self.layer._redo_history:
            return None
        history = self.redo_items.pop() if self.redo_items else None
        result = self.native_redo()
        if history is not None:
            self.widget._restore_overlap_history(history, undoing=False)
            self.undo_items.append(history)
        return result


def _atomic_save_overlap_store(
    store: OverlapStore,
    destination: Path,
    output_dtype: np.dtype,
    *,
    expected_identity: Optional[FileIdentity],
    require_absent: bool = False,
) -> SaveResult:
    """Project and embed one UI-locked overlap store in a worker thread."""
    projection = store.projection_view
    if projection.dtype != np.dtype(output_dtype):
        projection = np.ascontiguousarray(projection, dtype=output_dtype)
    payload = store.to_payload(projection)
    return atomic_save_labels(
        projection,
        destination,
        output_dtype,
        expected_identity=expected_identity,
        require_absent=require_absent,
        annotation_payload=payload,
        _copy_snapshot=False,
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
        self._labels_destination_identity: Optional[FileIdentity] = None
        self._mapping_destination_identity: Optional[FileIdentity] = None
        self._class_config_pending_save = False
        self._pending_deleted_value: Optional[int] = None
        self._pending_deleted_replacement: Optional[int] = None
        self._changing_selected_label = False
        self._save_worker = None
        self._active_save_path: Optional[Path] = None
        self._active_save_layer = None
        self._active_save_layer_mode = None
        self._active_save_adopt_destination = False
        self._active_save_update_class_config = False
        self._active_save_output_dtype: Optional[np.dtype] = None
        self._browse_buttons = []
        self._syncing_overlap_layer = False

        self.class_button_layout = None

        self.class_map: Dict[int, str] = {}
        self.class_colors: Dict[int, str] = {}

        self.labels_layer = None
        self.composite_layer = None
        self.overlap_store: Optional[OverlapStore] = None
        self.overlap_editor: Optional[OverlapEditorController] = None

        self._build_ui()
        self._bind_hotkeys()
        # napari recreates every layer's polygon overlay when layers are added
        # or removed. Reinstall our preview callback after that core rebuild.
        self.viewer.layers.events.inserted.connect(
            self._restore_fast_rendering_after_layer_change
        )
        self.viewer.layers.events.removed.connect(
            self._restore_fast_rendering_after_layer_change
        )

    def _restore_fast_rendering_after_layer_change(self, event=None):
        del event
        layer = self.labels_layer
        if layer is None or not any(
            candidate is layer for candidate in self.viewer.layers
        ):
            self._update_project_controls()
            return
        enable_fast_rendering(self.viewer, layer)
        composite = self.composite_layer
        if composite is not None and any(
            candidate is composite for candidate in self.viewer.layers
        ):
            enable_fast_texture_updates(self.viewer, composite)
        self._update_project_controls()

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

        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load_data)
        layout.addWidget(self.load_btn)

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

        self.delete_class_btn = QPushButton("Delete selected…")
        self.delete_class_btn.setEnabled(False)
        self.delete_class_btn.clicked.connect(self._delete_selected_class)
        self.delete_class_btn.setToolTip(
            "Delete the selected class and optionally reassign its pixels"
        )
        layout.addWidget(self.delete_class_btn)
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

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Overlay opacity"))
        self.overlay_opacity_slider = QSlider(Qt.Horizontal)
        self.overlay_opacity_slider.setRange(0, 100)
        self.overlay_opacity_slider.setValue(45)
        self.overlay_opacity_slider.setEnabled(False)
        self.overlay_opacity_slider.setToolTip(
            "Adjust the visible semantic annotation overlay"
        )
        self.overlay_opacity_slider.valueChanged.connect(
            self._set_overlay_opacity
        )
        opacity_row.addWidget(self.overlay_opacity_slider)
        self.overlay_opacity_value = QLabel("45%")
        opacity_row.addWidget(self.overlay_opacity_value)
        layout.addLayout(opacity_row)

        self.save_destination_caption = QLabel("Current save destination")
        layout.addWidget(self.save_destination_caption)
        self.save_destination_line = QLineEdit()
        self.save_destination_line.setReadOnly(True)
        self.save_destination_line.setPlaceholderText("No label file loaded")
        self.save_destination_line.setToolTip(
            "Save writes here while the loaded file is unchanged. Save As "
            "creates or replaces a destination you explicitly choose."
        )
        layout.addWidget(self.save_destination_line)

        self.save_btn = QPushButton("Save [s]")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_labels)
        layout.addWidget(self.save_btn)

        self.save_as_btn = QPushButton("Save As…")
        self.save_as_btn.setEnabled(False)
        self.save_as_btn.clicked.connect(self.save_labels_as)
        self.save_as_btn.setToolTip(
            "Save all current annotations and overlaps to another PNG or TIFF"
        )
        layout.addWidget(self.save_as_btn)

        self.undo_btn = QPushButton("Undo [u]")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self.undo)
        layout.addWidget(self.undo_btn)

        layout.addWidget(QLabel(
            "Usage: select a class, then use napari's native Paint, Fill, "
            "Erase, or Polygon tools. Add classes above; use Edit selected "
            "or right-click a class to rename or recolor it. Delete selected "
            "stages a class removal until Save."
        ))

        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(260)
        self.resize(320, self.height())

        for line_edit in (
            self.image_line,
            self.label_line,
            self.mapping_line,
        ):
            line_edit.textChanged.connect(self._update_project_controls)
        self._update_project_controls()

    def _file_row(self, line_edit, callback):
        row = QHBoxLayout()
        browse = QPushButton("Browse")
        browse.clicked.connect(callback)
        self._browse_buttons.append(browse)
        row.addWidget(line_edit)
        row.addWidget(browse)
        return row

    @staticmethod
    def _canonical_existing_file(value: str, description: str) -> Path:
        """Return one existing file as an absolute, symlink-resolved path."""
        text = str(value).strip()
        if not text:
            raise ValueError(f"Choose a {description.lower()} first.")

        candidate = Path(text).expanduser()
        try:
            path = candidate.resolve(strict=True)
        except OSError as error:
            raise ValueError(
                f"{description} does not exist or cannot be accessed: "
                f"{candidate}"
            ) from error
        if not path.is_file():
            raise ValueError(f"{description} is not a file: {path}")
        return path

    def _labels_layer_is_active(self) -> bool:
        edit_layer = self.labels_layer
        composite_layer = self.composite_layer
        if (
            edit_layer is None
            or composite_layer is None
            or self.overlap_editor is None
            or self.overlap_store is None
        ):
            return False
        return all(
            any(candidate is layer for candidate in self.viewer.layers)
            for layer in (edit_layer, composite_layer)
        )

    def _project_inputs_match_loaded(self) -> bool:
        if (
            self.image_path is None
            or self.labels_path is None
            or self.mapping_path is None
        ):
            return False
        return (
            self.image_line.text().strip() == str(self.image_path)
            and self.label_line.text().strip() == str(self.labels_path)
            and self.mapping_line.text().strip() == str(self.mapping_path)
        )

    @staticmethod
    def _destination_identity_matches(
        path: Path,
        expected_identity: Optional[FileIdentity],
    ) -> bool:
        if expected_identity is None:
            return False
        try:
            return (
                path.resolve(strict=True) == path
                and file_identity(path) == expected_identity
            )
        except (OSError, RuntimeError):
            return False

    def _update_project_controls(self, _text=None) -> None:
        """Keep actions synchronized with the successfully loaded project."""
        busy = self._save_worker is not None
        active_layer = self._labels_layer_is_active()
        inputs_match = self._project_inputs_match_loaded()
        has_destination = self.labels_path is not None

        self.load_btn.setEnabled(not busy)
        for line_edit in (
            self.image_line,
            self.label_line,
            self.mapping_line,
        ):
            line_edit.setEnabled(not busy)
        for browse_button in self._browse_buttons:
            browse_button.setEnabled(not busy)

        project_actions_enabled = active_layer and inputs_match and not busy
        destination_available = (
            has_destination
            and self.labels_path.is_absolute()
            and self._destination_identity_matches(
                self.labels_path,
                self._labels_destination_identity,
            )
        )
        mapping_available = (
            self.mapping_path is not None
            and self.mapping_path.is_absolute()
            and self._destination_identity_matches(
                self.mapping_path,
                self._mapping_destination_identity,
            )
        )
        # The live overlap store is the user's work.  A missing/replaced disk
        # target must never strand it by disabling Save: save_labels() routes
        # unsafe destinations through the same guarded Save As flow.
        self.save_btn.setEnabled(active_layer and not busy)
        self.save_as_btn.setEnabled(active_layer and not busy)
        rescue_required = not destination_available or (
            self._class_config_pending_save and not mapping_available
        )
        if active_layer and rescue_required:
            self.save_btn.setText("Save As… [s]")
        elif self._class_config_pending_save:
            self.save_btn.setText("Save class deletion [s]")
        else:
            self.save_btn.setText("Save [s]")
        self.undo_btn.setEnabled(active_layer and not busy)
        class_actions_enabled = (
            project_actions_enabled
            and destination_available
            and mapping_available
            and not self._class_config_pending_save
        )
        self.add_class_btn.setEnabled(class_actions_enabled)
        self.edit_class_btn.setEnabled(class_actions_enabled)
        selected_value = (
            int(self.overlap_editor.active_class or 0)
            if active_layer
            else 0
        )
        self.delete_class_btn.setEnabled(
            class_actions_enabled
            and selected_value > 0
            and selected_value in self.class_map
        )
        # A pending deletion blocks further class-definition changes until it
        # is saved, but class selection and painting remain available.
        self.class_button_container.setEnabled(project_actions_enabled)
        self.overlay_opacity_slider.setEnabled(project_actions_enabled)

        if self.labels_path is None:
            self.save_destination_line.clear()
            self.save_destination_caption.setText("Current save destination")
            self.save_destination_line.setToolTip(
                "Load a project before saving labels."
            )
            return

        destination = str(self.labels_path)
        self.save_destination_line.setText(destination)
        self.save_destination_line.setCursorPosition(0)
        self.save_destination_line.setToolTip(destination)
        if busy:
            self.save_destination_caption.setText("Saving labels to")
        elif not active_layer:
            self.save_destination_caption.setText(
                "Save disabled — loaded Labels layer was removed"
            )
        elif not destination_available:
            self.save_destination_caption.setText(
                "Original is missing or changed — Save opens Save As"
            )
        elif self._class_config_pending_save and not mapping_available:
            self.save_destination_caption.setText(
                "Class CSV changed — Save As preserves labels without it"
            )
        elif self._class_config_pending_save:
            self.save_destination_caption.setText(
                "Class deletion pending — Save writes labels and classes to"
            )
        elif not inputs_match:
            self.save_destination_caption.setText(
                "New paths not loaded — Save still writes to"
            )
        else:
            self.save_destination_caption.setText("Current save destination")

    def _dialog_start_directory(self, *line_edits: QLineEdit) -> str:
        """Start file pickers beside the current project when possible."""
        for line_edit in line_edits:
            text = line_edit.text().strip()
            if not text:
                continue
            candidate = Path(text).expanduser()
            directory = candidate if candidate.is_dir() else candidate.parent
            if directory.is_dir():
                return str(directory.resolve())
        if self.image_path is not None:
            return str(self.image_path.parent)
        return ""

    def _choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose histology image",
            self._dialog_start_directory(self.image_line),
        )
        if path:
            self.image_line.setText(path)

    def _choose_labels(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose label image",
            self._dialog_start_directory(self.label_line, self.image_line),
        )
        if path:
            self.label_line.setText(path)

    def _choose_mapping(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose class mapping CSV",
            self._dialog_start_directory(self.mapping_line, self.image_line),
        )
        if path:
            self.mapping_line.setText(path)

    def _bind_hotkeys(self):
        @self.viewer.bind_key("s", overwrite=True)
        def _save(viewer):
            self.save_labels()

        @self.viewer.bind_key("u", overwrite=True)
        def _undo(viewer):
            self.undo()

    @staticmethod
    def _overlap_store_for_project(
        labels: np.ndarray,
        labels_path: Path,
        class_map: Dict[int, str],
        class_colors: Dict[int, str],
    ) -> OverlapStore:
        """Load an embedded overlap model or losslessly migrate legacy data."""
        payload = read_embedded_annotations(labels_path)
        if payload is None:
            compact = LabelEditorWidget._compact_labels(labels, class_map)
            return OverlapStore.from_legacy(
                compact,
                class_map,
                class_colors,
            )

        # Embedded membership is authoritative.  The CSV remains the user's
        # editable class-definition file, so its names/colors override the
        # embedded copies while embedded-only classes are retained to avoid
        # silently discarding hidden annotations.
        store = OverlapStore.from_payload(payload, projection=labels)
        for value, name in sorted(class_map.items()):
            value = int(value)
            color = class_colors.get(value)
            if value == 0 or value in store.class_values:
                store.set_class_metadata(value, name=name, color=color)
            else:
                store.add_class(value, name, color, top=True)
        return store

    def load_data(self):
        if self._save_worker is not None:
            self.viewer.status = (
                "Wait for the current label save to finish before loading."
            )
            return

        # Preflight the complete candidate project in local variables.  No
        # active path or layer changes unless every file reads and validates
        # successfully.  A failed Load therefore cannot redirect a later Save
        # of the still-visible previous layer.
        image_path = self._canonical_existing_file(
            self.image_line.text(),
            "Histology image",
        )
        labels_path = self._canonical_existing_file(
            self.label_line.text(),
            "Label image",
        )
        mapping_path = self._canonical_existing_file(
            self.mapping_line.text(),
            "Class mapping CSV",
        )
        if labels_path.suffix.lower() not in {".png", ".tif", ".tiff"}:
            raise ValueError(
                "Overlapping annotations require a PNG or TIFF label image "
                "so their lossless data can stay inside the same file."
            )
        labels_destination_identity = file_identity(labels_path)
        mapping_destination_identity = file_identity(mapping_path)

        image = iio.imread(image_path)
        labels = iio.imread(labels_path)

        if labels.ndim != 2:
            raise ValueError(f"Label image must be 2D. Got {labels.shape}")

        if image.shape[:2] != labels.shape:
            raise ValueError(
                f"Image and labels differ: {image.shape[:2]} vs {labels.shape}"
            )

        class_map, class_colors = read_class_config(mapping_path)
        if not self._destination_identity_matches(
            labels_path,
            labels_destination_identity,
        ):
            raise RuntimeError(
                "The label image changed while it was loading. Nothing was "
                "activated; click Load to try again."
            )
        if not self._destination_identity_matches(
            mapping_path,
            mapping_destination_identity,
        ):
            raise RuntimeError(
                "The class mapping CSV changed while it was loading. Nothing "
                "was activated; click Load to try again."
            )
        labels_output_dtype = np.dtype(labels.dtype)
        overlap_store = self._overlap_store_for_project(
            labels,
            labels_path,
            class_map,
            class_colors,
        )
        class_map = overlap_store.class_map
        class_colors = overlap_store.class_colors
        self._validate_class_ids_for_destination(
            labels_path,
            class_map,
        )
        labels_output_dtype = self._promoted_dtype_for_classes(
            labels_output_dtype,
            class_map,
        )
        active_value = next(
            (value for value in sorted(overlap_store.class_values)),
            None,
        )
        overlap_editor = OverlapEditorController(
            overlap_store,
            active_value,
            projection_dtype=labels_output_dtype,
        )
        # The packed store and compact top projection are now authoritative.
        # Drop the separately decoded source mask before creating GPU layers;
        # on a whole-slide image this removes one full mask from peak load RAM.
        del labels

        is_rgb = image.ndim == 3 and image.shape[-1] in (3, 4)
        image_pyramid = build_image_pyramid(image) if is_rgb else [image]
        image_data = image_pyramid if len(image_pyramid) > 1 else image
        previous_layers = list(self.viewer.layers)
        previous_layer_ids = {id(layer) for layer in previous_layers}

        try:
            # File/data validation is complete, so release the previous GPU
            # scene before uploading the new (potentially very large) images.
            # Retain the layer objects themselves so a rare napari scene-
            # construction failure can restore the previous project.
            for previous_layer in reversed(previous_layers):
                if any(
                    candidate is previous_layer
                    for candidate in self.viewer.layers
                ):
                    self.viewer.layers.remove(previous_layer)

            image_layer = self.viewer.add_image(
                image_data,
                name="histology",
                rgb=is_rgb,
                multiscale=len(image_pyramid) > 1,
                interpolation2d="linear",
            )
            composite_layer = self.viewer.add_labels(
                overlap_editor.composite,
                name="labels — all classes",
                opacity=self.overlay_opacity_slider.value() / 100.0,
                features=self._label_features(class_map),
            )
            composite_layer.colormap = self._multiclass_colormap(
                class_map,
                class_colors=class_colors,
            )
            composite_layer.contour = 0
            composite_layer.editable = False

            labels_layer = self.viewer.add_labels(
                overlap_editor.edit_mask,
                name=self._active_layer_name(active_value, class_map),
                # The binary layer is a lightweight tool target only. The
                # all-class composite below it is the single visible labels
                # texture and receives bounded projection updates.
                opacity=0.0,
                features=self._active_edit_features(active_value, class_map),
            )
            labels_layer.colormap = self._active_edit_colormap(
                active_value,
                class_colors,
            )
            labels_layer.contour = 0
            labels_layer.selected_label = 1 if active_value is not None else 0
            labels_layer.n_edit_dimensions = 2
            labels_layer.contiguous = True
            labels_layer.preserve_labels = False
            self._limit_undo_history(labels_layer)
            enable_overlap_fill(
                labels_layer,
                overlap_editor.composite,
                active_value,
            )
            enable_fast_polygon(labels_layer)
            enable_fast_rendering(self.viewer, labels_layer)
            enable_fast_texture_updates(self.viewer, composite_layer)
        except BaseException:
            # An add-layer callback may insert a layer and then raise before
            # viewer.add_* returns. Identify rollback targets from the original
            # scene snapshot rather than only from successfully returned
            # objects, then restore the exact previous layer objects.
            cleanup_errors = []
            for candidate in reversed(list(self.viewer.layers)):
                if id(candidate) not in previous_layer_ids:
                    try:
                        self.viewer.layers.remove(candidate)
                    except Exception as cleanup_error:
                        cleanup_errors.append(cleanup_error)
            for previous_layer in previous_layers:
                if not any(
                    candidate is previous_layer
                    for candidate in self.viewer.layers
                ):
                    try:
                        self.viewer.layers.append(previous_layer)
                    except Exception as cleanup_error:
                        cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                self.viewer.status = (
                    "Load failed and napari could not completely restore the "
                    f"previous scene: {cleanup_errors[0]}"
                )
            raise

        self.image_path = image_path
        self.labels_path = labels_path
        self.mapping_path = mapping_path
        self.class_map = class_map
        self.class_colors = class_colors
        self._labels_output_dtype = labels_output_dtype
        self._labels_destination_identity = labels_destination_identity
        self._mapping_destination_identity = mapping_destination_identity
        self._class_config_pending_save = False
        self._pending_deleted_value = None
        self._pending_deleted_replacement = None
        self.overlap_store = overlap_store
        self.overlap_editor = overlap_editor
        self.composite_layer = composite_layer
        self.labels_layer = labels_layer
        self.labels_layer.events.selected_label.connect(
            self._on_selected_label_change
        )
        if hasattr(self.labels_layer.events, "labels_update"):
            self.labels_layer.events.labels_update.connect(
                self._on_overlap_labels_update
            )
        self.labels_layer.events.opacity.connect(
            self._on_active_layer_opacity_change
        )
        self._enable_overlap_edit_tracking()
        self._install_semantic_pick()
        self._install_semantic_tooltip()

        self.viewer.tooltip.visible = True

        self.image_line.setText(str(image_path))
        self.label_line.setText(str(labels_path))
        self.mapping_line.setText(str(mapping_path))
        self._populate_class_buttons()
        self._update_project_controls()

        self.viewer.status = (
            "Loaded lossless overlapping annotations. Save writes to "
            f"{self.labels_path}; Save As can create a rescue copy."
        )

    @staticmethod
    def _active_layer_name(
        value: Optional[int],
        class_map: Dict[int, str],
    ) -> str:
        if value is None:
            return "active class — none"
        return f"active class — {value}: {class_map.get(value, value)}"

    @staticmethod
    def _active_edit_features(
        value: Optional[int],
        class_map: Dict[int, str],
    ) -> pd.DataFrame:
        active_name = (
            "No active class"
            if value is None
            else f"{value} — {class_map.get(value, value)}"
        )
        return pd.DataFrame(
            {
                "index": [0, 1],
                "Label": ["Erase active class", active_name],
            }
        )

    def _active_edit_colormap(
        self,
        value: Optional[int],
        class_colors: Optional[Dict[int, str]] = None,
    ) -> DirectLabelColormap:
        active_color = (
            np.array([0.0, 0.0, 0.0, 0.0])
            if value is None
            else self._label_rgba(value, class_colors=class_colors)
        )
        return DirectLabelColormap(
            color_dict={
                None: np.array([0.0, 0.0, 0.0, 0.0]),
                0: np.array([0.0, 0.0, 0.0, 0.0]),
                1: active_color,
            }
        )

    def _refresh_overlap_layer_metadata(
        self,
        *,
        include_composite: bool = True,
        update_active_colormap: bool = True,
    ) -> None:
        if not self._labels_layer_is_active():
            return
        active = self.overlap_editor.active_class
        if include_composite:
            self.composite_layer.features = self._label_features(self.class_map)
            self.composite_layer.colormap = self._multiclass_colormap(
                self.class_map,
                class_colors=self.class_colors,
            )
        self.labels_layer.name = self._active_layer_name(
            active,
            self.class_map,
        )
        self.labels_layer.features = self._active_edit_features(
            active,
            self.class_map,
        )
        if update_active_colormap:
            self.labels_layer.colormap = self._active_edit_colormap(
                active,
                self.class_colors,
            )
        elif active is not None:
            # The 0/1 texture encoding is unchanged, but napari's colormap
            # setter normally reslices the complete hidden slide. Suppress
            # only that refresh while retaining LUT, cursor swatch, and
            # polygon-preview color events.
            layer = self.labels_layer
            had_instance_refresh = "refresh" in layer.__dict__
            previous_refresh = layer.__dict__.get("refresh")
            layer.refresh = lambda *args, **kwargs: None
            try:
                layer.colormap = self._active_edit_colormap(
                    active,
                    self.class_colors,
                )
            finally:
                if had_instance_refresh:
                    layer.refresh = previous_refresh
                else:
                    del layer.__dict__["refresh"]
        enable_overlap_fill(
            self.labels_layer,
            self.overlap_editor.composite,
            active,
        )
        self.labels_layer.preserve_labels = False

    def _on_overlap_labels_update(self, event=None) -> None:
        """Commit one napari partial edit into the packed membership store."""
        if (
            self._syncing_overlap_layer
            or self._save_worker is not None
            or event is None
            or not self._labels_layer_is_active()
        ):
            return
        try:
            update = np.asarray(event.data)
            dimensions = len(tuple(event.offset))
            shape = tuple(int(size) for size in update.shape[:dimensions])
            if dimensions != 2 or len(shape) != 2:
                raise ValueError("Only 2-D label edits are supported.")
            changed = self.overlap_editor.process_changed_patch(
                event.offset,
                shape,
            )
            if changed:
                self._refresh_composite_patch(event.offset, shape)
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            # Keep the UI alive if a future napari release changes its partial
            # event payload. Save/Undo perform a complete packed sync too.
            self.viewer.status = f"Could not synchronize label edit: {error}"

    def _refresh_composite_patch(self, offset, shape) -> None:
        """Upload one updated composite rectangle through napari's fast path."""
        if self.composite_layer is None:
            return
        row_start, column_start = (int(value) for value in offset)
        height, width = (int(value) for value in shape)
        updated_slice = (
            slice(row_start, row_start + height),
            slice(column_start, column_start + width),
        )
        try:
            raw = self.composite_layer._slice.image.raw
            encoded = self.composite_layer._raw_to_displayed(
                raw,
                data_slice=updated_slice,
            )
            cached_view = self.composite_layer._slice.image.view
            try:
                cached_view[updated_slice] = encoded
            except ValueError:
                # uint8 direct-label views can alias the read-only raw array.
                # The store mutation has already changed those shared bytes;
                # assignment is unnecessary when the encoded values match.
                if not np.array_equal(cached_view[updated_slice], encoded):
                    raise
            # Emit the same bounded event as Labels._partial_labels_refresh.
            # Calling that private method would redundantly assign the cached
            # view to itself, which fails when uint8 direct colors alias our
            # read-only authoritative projection.
            self.composite_layer.events.labels_update(
                data=encoded,
                offset=[row_start, column_start],
            )
            self.composite_layer._updated_slice = None
        except (AttributeError, IndexError, TypeError, ValueError):
            self.composite_layer._updated_slice = None
            self.composite_layer.refresh()

    def _refresh_composite_bounds(self, bounds) -> None:
        if bounds is None:
            return
        row_start, row_stop, column_start, column_stop = bounds
        self._refresh_composite_patch(
            (row_start, column_start),
            (row_stop - row_start, column_stop - column_start),
        )

    def _enable_overlap_edit_tracking(self) -> None:
        layer = self.labels_layer
        current = getattr(layer, "_napari_histo_edit_tracker", None)
        if current is not None and current.widget is self:
            return
        _OverlapEditTracker(self, layer).install()

    def _restore_overlap_history(self, history, *, undoing: bool) -> None:
        """Restore membership and per-pixel visible tops for one action."""
        self._syncing_overlap_layer = True
        try:
            self.overlap_editor.full_sync()
            atoms = reversed(history) if undoing else history
            combined = None
            for indices, top_before, top_after in atoms:
                values = top_before if undoing else top_after
                _changed, bounds = self.overlap_editor.restore_projection_indices(
                    indices,
                    values,
                )
                if bounds is None:
                    continue
                if combined is None:
                    combined = bounds
                else:
                    combined = (
                        min(combined[0], bounds[0]),
                        max(combined[1], bounds[1]),
                        min(combined[2], bounds[2]),
                        max(combined[3], bounds[3]),
                    )
            self._refresh_composite_bounds(combined)
        finally:
            self._syncing_overlap_layer = False

    def _install_semantic_pick(self) -> None:
        """Make Pick select the visible semantic class, not binary 0/1."""
        layer = self.labels_layer
        layer._drag_modes = dict(layer._drag_modes)

        def semantic_pick(_layer, event):
            value = self.composite_layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            self._select_label(0 if value is None else int(value))

        layer._napari_histo_semantic_pick = semantic_pick
        layer._drag_modes[Mode.PICK] = semantic_pick

    def _install_semantic_tooltip(self) -> None:
        """Report the visible semantic label through the binary edit layer."""
        layer = self.labels_layer

        def semantic_tooltip(
            _layer,
            position,
            *,
            view_direction=None,
            dims_displayed=None,
            world=False,
        ):
            return self._semantic_tooltip_text(
                position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=world,
            )

        layer._napari_histo_semantic_tooltip = semantic_tooltip
        layer._get_tooltip_text = MethodType(semantic_tooltip, layer)

    def _semantic_tooltip_text(
        self,
        position,
        *,
        view_direction=None,
        dims_displayed=None,
        world=False,
    ) -> str:
        composite = self.composite_layer
        value = composite.get_value(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        if value is None:
            return ""
        top = int(value)
        top_text = f"{top} — {self.class_map.get(top, top)}"
        if top == 0:
            return f"Label: {top_text}"

        coordinate = np.asarray(position)
        if world:
            coordinate = np.asarray(composite.world_to_data(coordinate))
        coordinate = np.round(coordinate).astype(int)
        displayed = tuple(composite._slice_input.displayed)
        if composite.ndim < len(coordinate):
            offset = len(coordinate) - composite.ndim
            coordinate = coordinate[
                [dimension + offset for dimension in displayed]
            ]
        else:
            coordinate = coordinate[list(displayed)]
        if len(coordinate) != 2 or np.any(coordinate < 0) or np.any(
            coordinate >= np.asarray(self.overlap_store.shape)
        ):
            return f"Label: {top_text}"

        memberships = self.overlap_store.memberships_at(*coordinate)
        if len(memberships) <= 1:
            return f"Label: {top_text}"
        membership_text = ", ".join(
            f"{member} — {self.class_map.get(member, member)}"
            for member in memberships
        )
        return f"Top: {top_text} | Memberships: {membership_text}"

    def _set_overlay_opacity(self, value: int) -> None:
        value = int(value)
        self.overlay_opacity_value.setText(f"{value}%")
        if self.composite_layer is not None and any(
            layer is self.composite_layer for layer in self.viewer.layers
        ):
            self.composite_layer.opacity = value / 100.0

    def _on_active_layer_opacity_change(self, event=None) -> None:
        del event
        if self.labels_layer is None or self.labels_layer.opacity == 0:
            return
        self.labels_layer.opacity = 0.0
        self.viewer.status = (
            "Use the plugin's Overlay opacity slider for visible annotations."
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
        if self._save_worker is not None:
            self.viewer.status = (
                "Wait for the current label save to finish before switching "
                "classes."
            )
            return
        if not self._labels_layer_is_active():
            return

        value = int(value)
        if value == 0:
            self.labels_layer.selected_label = 0
            active = self.overlap_editor.active_class
            self.viewer.layers.selection.active = self.labels_layer
            self.viewer.status = (
                "Erase active class"
                if active is None
                else f"Erase {active}: {self.class_map.get(active, active)}"
            )
            self._update_project_controls()
            return
        if value not in self.class_map or value not in self.overlap_store.class_values:
            self.viewer.status = f"Class {value} is not available."
            return

        # Flush any edit whose partial event was suppressed before replacing
        # the binary working mask with the newly selected membership plane.
        missed_changes = self.overlap_editor.full_sync()
        self._syncing_overlap_layer = True
        try:
            self.overlap_editor.select_class(value)
            self._limit_undo_history(self.labels_layer)
            self.labels_layer.selected_label = 1
            if missed_changes:
                self.composite_layer.refresh()
            self._refresh_overlap_layer_metadata(
                include_composite=False,
                update_active_colormap=False,
            )
        finally:
            self._syncing_overlap_layer = False
        self.viewer.layers.selection.active = self.labels_layer
        self.viewer.status = (
            f"Selected class {value}: {self.class_map.get(value, value)}"
        )
        self._update_project_controls()

    def _on_selected_label_change(self, event=None) -> None:
        """Keep the binary edit layer's numeric control at erase or paint."""
        del event
        if self._changing_selected_label or not self._labels_layer_is_active():
            return
        value = int(self.labels_layer.selected_label)
        valid = value in {0, 1} and (
            value == 0 or self.overlap_editor.active_class is not None
        )
        if not valid:
            fallback = 0
            self._changing_selected_label = True
            try:
                self.labels_layer.selected_label = fallback
            finally:
                self._changing_selected_label = False
            self.viewer.status = (
                f"The active edit layer accepts only 0 (erase) or 1 (paint); "
                f"selected {fallback}. Choose semantic classes with the "
                "class buttons."
            )
        self._update_project_controls()

    def _add_class(self) -> None:
        if self._save_worker is not None:
            self.viewer.status = "Wait for the current label save to finish."
            return
        if (
            not self._labels_layer_is_active()
            or self.mapping_path is None
            or not self._project_inputs_match_loaded()
        ):
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
        if self._save_worker is not None:
            self.viewer.status = "Wait for the current label save to finish."
            return
        if (
            not self._labels_layer_is_active()
            or not self._project_inputs_match_loaded()
        ):
            self.viewer.status = "Load labels first."
            return
        self._edit_class(int(self.overlap_editor.active_class or 0))

    def _edit_class(self, value: int) -> None:
        if self._save_worker is not None:
            self.viewer.status = "Wait for the current label save to finish."
            return
        if self._class_config_pending_save:
            self.viewer.status = (
                "Save the pending class deletion before editing classes."
            )
            return
        if (
            not self._labels_layer_is_active()
            or not self._project_inputs_match_loaded()
        ):
            self.viewer.status = "Click Load before editing classes."
            return
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

    def _delete_selected_class(self) -> None:
        if not self._labels_layer_is_active():
            self.viewer.status = "Load labels before deleting a class."
            return
        self._delete_class(int(self.overlap_editor.active_class or 0))

    def _delete_class(self, value: int) -> None:
        """Stage one class removal without writing either project file."""
        try:
            value = int(value)
            self._validate_class_deletion(value)
            pixel_count = self.overlap_store.count_class(value)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            self.viewer.status = f"Could not delete class: {error}"
            return

        dialog = DeleteClassDialog(
            value,
            self.class_map[value],
            pixel_count,
            self.class_map,
            parent=self,
        )
        if not self._execute_dialog(dialog):
            return

        try:
            # Revalidate after the modal confirmation. This protects against
            # external replacement of either locked destination while the
            # dialog was open.
            self._validate_class_deletion(value)
            replacement = dialog.replacement_value
            if replacement != 0 and replacement not in self.class_map:
                raise ValueError(
                    f"Replacement class {replacement} is no longer available."
                )

            deleted_name = self.class_map[value]
            changed = self.overlap_editor.delete_class(value, replacement)
            self.class_map = self.overlap_store.class_map
            self.class_colors = self.overlap_store.class_colors

            # Class deletion deliberately has no large sparse undo snapshot:
            # on a slide-sized mask, storing two int64 coordinate arrays can
            # consume several gigabytes. Resetting history also prevents an
            # older Undo operation from reintroducing the removed value.
            self._limit_undo_history(self.labels_layer)
            self._class_config_pending_save = True
            self._pending_deleted_value = value
            self._pending_deleted_replacement = replacement

            self._syncing_overlap_layer = True
            try:
                self.labels_layer.selected_label = (
                    1 if self.overlap_editor.active_class is not None else 0
                )
                self.labels_layer.refresh()
                self.composite_layer.refresh()
                self._refresh_overlap_layer_metadata()
            finally:
                self._syncing_overlap_layer = False
            self._populate_class_buttons()
            self.viewer.layers.selection.active = self.labels_layer
            self._update_project_controls()

            replacement_name = self.class_map.get(
                replacement,
                "Background" if replacement == 0 else str(replacement),
            )
            if changed:
                change_summary = (
                    f"; reassigned {changed:,} pixels to "
                    f"{replacement}: {replacement_name}"
                )
            else:
                change_summary = "; it was not used by any pixels"
            self.viewer.status = (
                f"Deleted class {value}: {deleted_name}{change_summary}. "
                "Press Save to write this change."
            )
        except (
            OSError,
            TypeError,
            ValueError,
            RuntimeError,
            MemoryError,
        ) as error:
            self.viewer.status = f"Could not delete class: {error}"
            QMessageBox.critical(self, "Could not delete class", str(error))

    def _validate_class_deletion(self, value: int) -> None:
        if self._save_worker is not None:
            raise ValueError("Wait for the current label save to finish.")
        if self._class_config_pending_save:
            raise ValueError(
                "Save the pending class deletion before deleting another class."
            )
        if not self._labels_layer_is_active():
            raise ValueError("The loaded Labels layer is no longer present.")
        if not self._project_inputs_match_loaded():
            raise ValueError(
                "Project paths changed; click Load before deleting classes."
            )
        if value == 0:
            raise ValueError("Background is reserved and cannot be deleted.")
        if value not in self.class_map:
            raise ValueError(f"Class value {value} is not in the class mapping.")
        if (
            self.labels_path is None
            or not self.labels_path.is_absolute()
            or not self._destination_identity_matches(
                self.labels_path,
                self._labels_destination_identity,
            )
        ):
            raise ValueError(
                "The loaded label image is missing or changed; nothing was "
                "deleted. Choose it again and click Load."
            )
        if (
            self.mapping_path is None
            or not self.mapping_path.is_absolute()
            or not self._destination_identity_matches(
                self.mapping_path,
                self._mapping_destination_identity,
            )
        ):
            raise ValueError(
                "The loaded class mapping CSV is missing or changed; nothing "
                "was deleted. Choose it again and click Load."
            )

    @staticmethod
    def _count_label_pixels(
        labels,
        value: int,
        *,
        mask_bytes: int = CLASS_SCAN_MASK_BYTES,
    ) -> int:
        """Count one label using a bounded reusable boolean buffer."""
        data = np.asarray(labels)
        if data.ndim != 2:
            raise ValueError("Class deletion requires a 2D label image.")
        if mask_bytes < 1:
            raise ValueError("mask_bytes must be at least 1.")
        height, width = data.shape
        if height == 0 or width == 0:
            return 0
        rows_per_chunk = max(1, min(height, mask_bytes // width))
        matches = np.empty((rows_per_chunk, width), dtype=bool)
        count = 0
        for row_start in range(0, height, rows_per_chunk):
            row_stop = min(height, row_start + rows_per_chunk)
            local_matches = matches[: row_stop - row_start]
            np.equal(data[row_start:row_stop], value, out=local_matches)
            count += int(np.count_nonzero(local_matches))
        return count

    @staticmethod
    def _reassign_label_pixels(
        labels,
        value: int,
        replacement: int,
        *,
        mask_bytes: int = CLASS_SCAN_MASK_BYTES,
    ) -> int:
        """Replace one label in-place with bounded temporary memory."""
        data = np.asarray(labels)
        if data.ndim != 2:
            raise ValueError("Class deletion requires a 2D label image.")
        if not data.flags.writeable:
            raise ValueError("The loaded label image is not editable.")
        if value == replacement:
            raise ValueError("A deleted class cannot replace itself.")
        if mask_bytes < 1:
            raise ValueError("mask_bytes must be at least 1.")
        height, width = data.shape
        if height == 0 or width == 0:
            return 0
        rows_per_chunk = max(1, min(height, mask_bytes // width))
        matches = np.empty((rows_per_chunk, width), dtype=bool)
        changed = 0
        for row_start in range(0, height, rows_per_chunk):
            row_stop = min(height, row_start + rows_per_chunk)
            chunk = data[row_start:row_stop]
            local_matches = matches[: row_stop - row_start]
            np.equal(chunk, value, out=local_matches)
            changed += int(np.count_nonzero(local_matches))
            np.putmask(chunk, local_matches, replacement)
        return changed

    @staticmethod
    def _execute_dialog(dialog) -> bool:
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
        except (
            OSError,
            TypeError,
            ValueError,
            RuntimeError,
            MemoryError,
        ) as error:
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
        if self._save_worker is not None:
            raise ValueError("Wait for the current label save to finish.")
        if self._class_config_pending_save:
            raise ValueError(
                "Save the pending class deletion before editing classes."
            )
        if self.labels_layer is not None and not self._labels_layer_is_active():
            raise ValueError("The loaded Labels layer is no longer present.")
        if (
            self.labels_layer is not None
            and not self._project_inputs_match_loaded()
        ):
            raise ValueError(
                "Project paths changed; click Load before editing classes."
            )

        value = int(value)
        name = str(name).strip()
        color = normalize_color(color)
        if value <= 0:
            raise ValueError("Class values must be positive; 0 is Background.")
        if not name:
            raise ValueError("Class name cannot be blank.")
        if color is None:
            raise ValueError("Choose a valid class color.")
        if self.labels_path is not None:
            # PNG has no safe wider integer representation. Refuse before
            # writing the CSV or mutating the packed store/UI so the project
            # can never enter a state that its label destination cannot save.
            self._validate_class_ids_for_destination(
                self.labels_path,
                (value,),
            )

        new_class_map = dict(self.class_map)
        new_class_map[value] = name
        new_class_colors = dict(self.class_colors)
        new_class_colors[value] = color

        promoted_output_dtype = self._labels_output_dtype
        if self.overlap_store is not None:
            output_dtype = (
                self.overlap_store.projection_dtype
                if self._labels_output_dtype is None
                else self._labels_output_dtype
            )
            promoted_output_dtype = self._promoted_dtype_for_classes(
                output_dtype,
                new_class_map,
            )
        elif self._labels_output_dtype is not None:
            promoted_output_dtype = self._promoted_dtype_for_classes(
                self._labels_output_dtype,
                new_class_map,
            )

        updated_mapping_identity = self._mapping_destination_identity
        if persist:
            if self.mapping_path is None:
                raise ValueError("No class mapping CSV is loaded.")
            if (
                not self.mapping_path.is_absolute()
                or not self._destination_identity_matches(
                    self.mapping_path,
                    self._mapping_destination_identity,
                )
            ):
                raise ValueError(
                    "The loaded class mapping CSV is missing or changed; "
                    "nothing was written. Choose it again and click Load."
                )
            saved_mapping_path = atomic_write_class_config(
                self.mapping_path,
                new_class_map,
                new_class_colors,
                expected_identity=self._mapping_destination_identity,
            )
            updated_mapping_identity = file_identity(saved_mapping_path)

        if self.overlap_store is not None:
            if value in self.overlap_store.class_values:
                self.overlap_store.set_class_metadata(
                    value,
                    name=name,
                    color=color,
                )
            else:
                self.overlap_store.add_class(value, name, color, top=True)
                if (
                    self.overlap_editor is not None
                    and self.overlap_editor.refresh_projection_reference()
                    and self.composite_layer is not None
                ):
                    self.composite_layer.data = self.overlap_editor.composite
                    self.composite_layer.editable = False
                    enable_fast_texture_updates(
                        self.viewer,
                        self.composite_layer,
                    )
            self.class_map = self.overlap_store.class_map
            self.class_colors = self.overlap_store.class_colors
        else:
            self.class_map = new_class_map
            self.class_colors = new_class_colors
        self._labels_output_dtype = promoted_output_dtype
        self._mapping_destination_identity = updated_mapping_identity

        if self._labels_layer_is_active():
            self._refresh_overlap_layer_metadata()

        self._populate_class_buttons()
        self._select_label(value)
        suffix = " and saved to the class CSV" if persist else ""
        self.viewer.status = f"Class {value}: {name} updated{suffix}."

    @staticmethod
    def _validate_class_ids_for_destination(
        labels_path: Path,
        class_values,
    ) -> None:
        """Reject class IDs that the selected image format cannot preserve."""
        if labels_path.suffix.lower() != ".png":
            return
        if any(
            int(value) > np.iinfo(np.uint16).max
            for value in class_values
        ):
            raise ValueError(
                "PNG supports class IDs up to 65535; use TIFF for larger "
                "class IDs."
            )

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

    def _label_rgba(
        self,
        value: int,
        *,
        class_colors: Optional[Dict[int, str]] = None,
    ):
        colors = self.class_colors if class_colors is None else class_colors
        configured = normalize_color(colors.get(int(value)))
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

    def _multiclass_colormap(self, class_map, *, class_colors=None):
        color_dict = {
            None: np.array([0, 0, 0, 0]),
            0: np.array([0, 0, 0, 0]),
        }

        for value in sorted(class_map):
            if value == 0:
                continue
            color_dict[int(value)] = self._label_rgba(
                int(value),
                class_colors=class_colors,
            )

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
        tracker = getattr(layer, "_napari_histo_edit_tracker", None)
        if tracker is not None:
            tracker.undo_items = deque(maxlen=int(limit))
            tracker.redo_items = deque(maxlen=int(limit))
            tracker.staged = []

    def undo(self):
        if self._save_worker is not None:
            self.viewer.status = "Wait for the current label save to finish."
            return
        if not self._labels_layer_is_active() or not self._undo_labels_layer(
            self.labels_layer
        ):
            self.viewer.status = "Nothing to undo."
            return

        try:
            missed_changes = self.overlap_editor.full_sync()
            if missed_changes:
                self.composite_layer.refresh()
        except (TypeError, ValueError, RuntimeError, MemoryError) as error:
            self.viewer.status = f"Undo display changed but sync failed: {error}"
            return
        self.viewer.status = "Undo complete."

    def save_labels(self):
        """Save normally, or rescue to Save As when a target is unsafe."""
        self._begin_label_save(force_save_as=False)

    def save_labels_as(self):
        """Save the complete live overlap model to a user-chosen image."""
        self._begin_label_save(force_save_as=True)

    def _choose_save_as_destination(
        self,
    ) -> Optional[tuple[Path, Optional[FileIdentity], bool]]:
        """Return a guarded Save As target, or ``None`` after cancellation."""
        if self.labels_path is not None:
            original = self.labels_path
            initial = original.with_name(
                f"{original.stem}-copy{original.suffix.lower()}"
            )
        else:
            initial = Path(
                self._dialog_start_directory(self.label_line, self.image_line)
            ) / "labels-copy.tif"

        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save annotations as",
            str(initial),
            "Label images (*.png *.tif *.tiff)",
        )
        if not selected:
            self.viewer.status = (
                "Save As cancelled; current annotations remain open and "
                "unsaved."
            )
            return None

        candidate = Path(selected).expanduser()
        if not candidate.suffix:
            candidate = candidate.with_suffix(".tif")
        if candidate.suffix.lower() not in {".png", ".tif", ".tiff"}:
            raise ValueError(
                "Save As requires a PNG, TIF, or TIFF label destination."
            )

        try:
            parent = candidate.parent.resolve(strict=True)
        except OSError as error:
            raise ValueError(
                f"Save As folder does not exist or cannot be accessed: "
                f"{candidate.parent}"
            ) from error
        if not parent.is_dir():
            raise ValueError(f"Save As parent is not a folder: {parent}")
        destination = parent / candidate.name

        if destination.is_symlink() or destination.exists():
            try:
                destination = destination.resolve(strict=True)
            except OSError as error:
                raise ValueError(
                    f"Save As destination cannot be accessed: {destination}"
                ) from error
            if not destination.is_file():
                raise ValueError(
                    f"Save As destination is not a file: {destination}"
                )
            expected_identity = file_identity(destination)
            require_absent = False
        else:
            expected_identity = None
            require_absent = True

        # Compare only canonical destinations, including an existing symlink's
        # referent.  A Save As alias must never bypass project-file or stale
        # original protections.
        if (
            self.labels_path is not None
            and destination == self.labels_path
            and not self._destination_identity_matches(
                self.labels_path,
                self._labels_destination_identity,
            )
        ):
            raise ValueError(
                "The original label destination is missing or changed. "
                "Choose a different filename so it is not overwritten."
            )
        protected_destinations = (
            (self.image_path, "loaded histology image"),
            (self.mapping_path, "loaded class mapping CSV"),
        )
        for protected_path, description in protected_destinations:
            if protected_path is not None and destination == protected_path:
                raise ValueError(
                    f"Save As cannot overwrite the {description}: "
                    f"{protected_path}"
                )

        self._validate_class_ids_for_destination(
            destination,
            self.class_map,
        )
        return destination, expected_identity, require_absent

    def _save_output_dtype_for_destination(
        self,
        destination: Path,
    ) -> np.dtype:
        """Choose a lossless disk dtype compatible with the chosen format."""
        if destination.suffix.lower() == ".png":
            return self._promoted_dtype_for_classes(
                np.dtype(np.uint8),
                self.class_map,
            )
        current = (
            self.overlap_store.projection_dtype
            if self._labels_output_dtype is None
            else self._labels_output_dtype
        )
        return self._promoted_dtype_for_classes(current, self.class_map)

    def _begin_label_save(self, *, force_save_as: bool) -> None:
        if self._save_worker is not None:
            self.viewer.status = "A label save is already running."
            return

        if self.labels_path is None or self.labels_layer is None:
            self.viewer.status = "No labels loaded."
            return

        if not self._labels_layer_is_active():
            self.viewer.status = (
                "Save disabled: the loaded Labels layer was removed."
            )
            self._update_project_controls()
            return

        normal_destination_available = (
            self.labels_path.is_absolute()
            and self._destination_identity_matches(
                self.labels_path,
                self._labels_destination_identity,
            )
        )
        mapping_available = not self._class_config_pending_save or (
            self.mapping_path is not None
            and (
                self.mapping_path.is_absolute()
                and self._destination_identity_matches(
                    self.mapping_path,
                    self._mapping_destination_identity,
                )
            )
        )
        # A pending deletion with a missing/replaced mapping is also rescued
        # to a self-contained label image.  Save As never writes the suspect
        # CSV; its pending state remains visible for a later explicit retry.
        use_save_as = (
            force_save_as
            or not normal_destination_available
            or not mapping_available
        )
        if use_save_as:
            try:
                chosen = self._choose_save_as_destination()
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                self.viewer.status = f"Save As failed: {error}"
                QMessageBox.critical(self, "Save As failed", str(error))
                self._update_project_controls()
                return
            if chosen is None:
                self._update_project_controls()
                return
            destination, expected_identity, require_absent = chosen
        else:
            destination = self.labels_path
            expected_identity = self._labels_destination_identity
            require_absent = False

        if self._class_config_pending_save:
            try:
                self._sanitize_pending_class_deletion()
            except (
                TypeError,
                ValueError,
                RuntimeError,
                MemoryError,
            ) as error:
                self.viewer.status = (
                    f"Save failed while validating the class deletion: {error}"
                )
                QMessageBox.critical(
                    self,
                    "Could not validate class deletion",
                    str(error),
                )
                self._update_project_controls()
                return

        try:
            self.overlap_editor.full_sync()
        except (TypeError, ValueError, RuntimeError, MemoryError) as error:
            self.viewer.status = f"Save failed while preparing annotations: {error}"
            QMessageBox.critical(self, "Save failed", str(error))
            return
        try:
            output_dtype = self._save_output_dtype_for_destination(destination)
        except (TypeError, ValueError) as error:
            self.viewer.status = f"Save failed: {error}"
            QMessageBox.critical(self, "Save failed", str(error))
            return
        saved_layer = self.labels_layer
        self._active_save_path = destination
        self._active_save_layer = saved_layer
        self._active_save_layer_mode = saved_layer.mode
        self._active_save_adopt_destination = use_save_as
        self._active_save_update_class_config = (
            self._class_config_pending_save and not use_save_as
        )
        self._active_save_output_dtype = np.dtype(output_dtype)
        # Labels.editable=False resets napari's native Undo queue. Pan/zoom
        # mode blocks drawing during the worker save while preserving sparse
        # binary and visible-top history for a later Undo.
        saved_layer.mode = Mode.PAN_ZOOM
        self._syncing_overlap_layer = True
        self.viewer.status = f"Saving labels to {destination}…"

        try:
            worker = create_worker(
                _atomic_save_overlap_store,
                self.overlap_store,
                destination,
                output_dtype,
                expected_identity=expected_identity,
                require_absent=require_absent,
                _start_thread=False,
                _ignore_errors=True,
            )
        except Exception as error:
            saved_layer.mode = self._active_save_layer_mode
            self._syncing_overlap_layer = False
            self._active_save_path = None
            self._active_save_layer = None
            self._active_save_layer_mode = None
            self._active_save_adopt_destination = False
            self._active_save_update_class_config = False
            self._active_save_output_dtype = None
            self._update_project_controls()
            self._on_save_error(error)
            return

        worker.returned.connect(self._on_save_complete)
        worker.errored.connect(self._on_save_error)
        worker.finished.connect(self._on_save_finished)
        # Own the worker before starting it. superqt only marks a worker as
        # running inside its thread, which otherwise allows rapid double-S
        # presses to launch two writes to the same file.
        self._save_worker = worker
        self._update_project_controls()
        try:
            worker.start()
        except Exception as error:
            self._save_worker = None
            saved_layer.mode = self._active_save_layer_mode
            self._syncing_overlap_layer = False
            self._active_save_path = None
            self._active_save_layer = None
            self._active_save_layer_mode = None
            self._active_save_adopt_destination = False
            self._active_save_update_class_config = False
            self._active_save_output_dtype = None
            self._update_project_controls()
            self._on_save_error(error)

    def _on_save_complete(self, result: SaveResult) -> None:
        saved_path = result.path.resolve(strict=False)
        if (
            self._active_save_path is not None
            and saved_path != self._active_save_path
        ):
            self._on_save_error(
                RuntimeError(
                    "The save worker returned an unexpected destination: "
                    f"{saved_path}"
                )
            )
            return
        if not self._destination_identity_matches(
            saved_path,
            result.identity,
        ):
            if not self._active_save_adopt_destination:
                self._labels_destination_identity = None
            self._on_save_error(
                RuntimeError(
                    "Labels were written, but the saved file could not be "
                    "verified because its destination changed immediately."
                )
            )
            return
        self._labels_destination_identity = result.identity
        if self._active_save_adopt_destination:
            self.labels_path = saved_path
            if self._active_save_output_dtype is not None:
                self._labels_output_dtype = self._active_save_output_dtype
            self.label_line.setText(str(saved_path))

        if (
            self._class_config_pending_save
            and self._active_save_update_class_config
        ):
            try:
                saved_mapping_path = self._write_pending_class_config()
            except (
                OSError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as error:
                self.viewer.status = (
                    f"Labels saved to {saved_path}, but the class CSV was "
                    f"not updated: {error}"
                )
                QMessageBox.critical(
                    self,
                    "Labels saved; class CSV not updated",
                    "The label image was saved safely, but the class mapping "
                    f"CSV was not changed:\n\n{error}\n\nIf it is still the "
                    "original CSV, its extra deleted-class row is harmless "
                    "because the saved labels no longer use that value. "
                    "Restore or reload the CSV before trying again.",
                )
                return
            self.viewer.status = (
                f"Saved labels to {saved_path} and class definitions to "
                f"{saved_mapping_path}"
            )
            return
        if self._class_config_pending_save:
            self.viewer.status = (
                f"Saved all labels and overlaps to {saved_path}. The class "
                "mapping CSV was not changed; its pending update remains."
            )
            return
        self.viewer.status = f"Saved labels to {saved_path}"

    def _write_pending_class_config(self) -> Path:
        """Persist staged class metadata after its label mask is safely saved."""
        if not self._class_config_pending_save:
            raise ValueError("No class definition changes are pending.")
        if self.mapping_path is None:
            raise ValueError("No class mapping CSV is loaded.")
        if (
            not self.mapping_path.is_absolute()
            or not self._destination_identity_matches(
                self.mapping_path,
                self._mapping_destination_identity,
            )
        ):
            raise ValueError(
                "The loaded class mapping CSV is missing or changed; it was "
                "not overwritten. Choose it again and click Load."
            )

        saved_mapping_path = atomic_write_class_config(
            self.mapping_path,
            self.class_map,
            self.class_colors,
            expected_identity=self._mapping_destination_identity,
        )
        updated_identity = file_identity(saved_mapping_path)
        if not self._destination_identity_matches(
            saved_mapping_path,
            updated_identity,
        ):
            raise RuntimeError(
                "The class mapping destination changed immediately after it "
                "was written. Click Load before changing classes again."
            )
        self._mapping_destination_identity = updated_identity
        self._class_config_pending_save = False
        self._pending_deleted_value = None
        self._pending_deleted_replacement = None
        return saved_mapping_path

    def _sanitize_pending_class_deletion(self) -> int:
        """Synchronize the active plane before saving a staged deletion."""
        if not self._class_config_pending_save:
            return 0
        if self._pending_deleted_value is None:
            raise RuntimeError("The pending deleted class value is missing.")
        if self._pending_deleted_replacement is None:
            raise RuntimeError("The pending replacement class is missing.")
        if not self._labels_layer_is_active():
            raise RuntimeError("The loaded Labels layer is no longer present.")

        replacement = int(self._pending_deleted_replacement)
        if replacement not in self.class_map:
            raise RuntimeError(
                f"Replacement class {replacement} is no longer available."
            )
        # Deleted semantic values cannot be typed into the binary napari
        # layer. A complete sync catches any edit event that was coalesced or
        # suppressed immediately before Save.
        return int(self.overlap_editor.full_sync())

    def _on_save_error(self, error: Exception) -> None:
        self.viewer.status = f"Save failed: {error}"
        QMessageBox.critical(self, "Save failed", str(error))

    def _on_save_finished(self) -> None:
        saved_layer = self._active_save_layer
        if saved_layer is not None and any(
            candidate is saved_layer for candidate in self.viewer.layers
        ):
            saved_layer.mode = self._active_save_layer_mode
        self._save_worker = None
        self._syncing_overlap_layer = False
        self._active_save_path = None
        self._active_save_layer = None
        self._active_save_layer_mode = None
        self._active_save_adopt_destination = False
        self._active_save_update_class_config = False
        self._active_save_output_dtype = None
        self._update_project_controls()
