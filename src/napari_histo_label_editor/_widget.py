from __future__ import annotations

import warnings

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
from napari.layers.labels._labels_utils import (
    mouse_event_to_labels_coordinate,
)
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QMessageBox, QGridLayout, 
    QSizePolicy, QScrollArea, QDockWidget
)
from qtpy.QtGui import QColor
from qtpy.QtCore import QTimer, Qt
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
from ._native_labels_controls import adapt_native_labels_tool_controls
from ._overlap_editor import OverlapEditorController
from ._overlap_store import OverlapDelta, OverlapStore
from ._object_selection import (
    AnnotationObjectSelection,
    SelectionTooLargeError,
    select_visible_component,
)

Image.MAX_IMAGE_PIXELS = None

MAX_UNDO_HISTORY = 20
CLASS_SCAN_MASK_BYTES = 8 * 1024 * 1024
MAX_OBJECT_MOVE_HISTORY_BYTES = 256 * 1024 * 1024
DEFAULT_ANNOTATION_OPACITY = 0.45
NATIVE_BRUSH_SIZE_MAX = 512
SELECTION_ACTIONS_DOCK_NAME = "Histology selected region"


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
        self.semantic_erase_seen: set[int] = set()
        self.semantic_erase_compact_done = False
        self.semantic_erase_dedup_exhausted = False

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
        self.semantic_erase_seen.clear()
        self.semantic_erase_compact_done = False
        self.semantic_erase_dedup_exhausted = False

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
        if self.widget._annotation_selection is not None:
            # Any pixel edit can split, merge, or retop the connected region.
            # Clear the preview before coordinates become stale.
            self.widget._clear_annotation_selection()
        controller = self.widget.overlap_editor
        normalized = controller.normalize_indices(indices)
        value_array = np.asarray(value)
        if (
            value_array.ndim == 0
            and int(value_array) == 0
            and not self.widget._syncing_overlap_layer
            and self.widget._labels_layer_is_active()
            and controller.active_class is not None
        ):
            return self._erase_visible_semantics(normalized, refresh)
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

    def _erase_visible_semantics(self, normalized, refresh=True):
        """Erase one visible class per touched pixel, independent of active."""

        rows, columns = normalized
        if rows.size == 0:
            return None
        new_linear = np.empty(0, dtype=np.intp)
        compact_region = self.layer._block_history and self.layer.mode in {
            Mode.FILL,
            Mode.POLYGON,
        }
        if self.layer._block_history:
            if self.semantic_erase_dedup_exhausted:
                return None
            if compact_region:
                # Napari gives each Fill click and completed Polygon its own
                # history block. Accept exactly one complete region in that
                # block: after revealing an underlying class, recomputing the
                # component can produce a different superset, so a digest of
                # the first region alone cannot prevent a second-level peel.
                # A boolean is also O(1) for a multi-million-pixel Fill.
                if self.semantic_erase_compact_done:
                    return None
            else:
                linear = rows * self.layer.data.shape[1] + columns
                unseen = np.fromiter(
                    (
                        int(index) not in self.semantic_erase_seen
                        for index in linear
                    ),
                    dtype=bool,
                    count=linear.size,
                )
                if not np.any(unseen):
                    return None
                rows = rows[unseen]
                columns = columns[unseen]
                normalized = (rows, columns)
                new_linear = linear[unseen]

        controller = self.widget.overlap_editor
        binary_before = np.array(
            self.layer.data[rows, columns],
            copy=True,
        )
        top_before = np.array(
            controller.composite[rows, columns],
            copy=True,
        )
        active = int(controller.active_class)
        active_top = top_before == active
        undo_before = tuple(self.layer._undo_history)
        redo_before = tuple(self.layer._redo_history)
        staged_before = tuple(self.layer._staged_history)
        updated_slice_before = self.layer._updated_slice
        history_before = self._native_history_marker()
        result = None
        try:
            if np.any(active_top):
                # Native Labels emits labels_update synchronously. Suppress
                # the ordinary active-plane adapter until the semantic store
                # removes each pixel's captured visible class; otherwise the
                # active subset could be projected twice before mixed-class
                # erasing completes.
                self.widget._syncing_overlap_layer = True
                try:
                    result = self.native_data_setitem(
                        (rows[active_top], columns[active_top]),
                        0,
                        refresh,
                    )
                finally:
                    self.widget._syncing_overlap_layer = False
            changed, bounds, erased_values, top_after = (
                controller.erase_visible_indices(normalized)
            )
        except BaseException:
            # A sparse store allocation can still fail on a memory-constrained
            # slide. Restore the binary proxy and napari queues exactly so a
            # reported failure never creates a half-native, half-semantic edit.
            self.layer.data[rows, columns] = binary_before
            self.layer._undo_history.clear()
            self.layer._undo_history.extend(undo_before)
            self.layer._redo_history.clear()
            self.layer._redo_history.extend(redo_before)
            self.layer._staged_history[:] = staged_before
            try:
                self.layer.refresh()
            except Exception:
                pass
            finally:
                self.layer._updated_slice = updated_slice_before
            raise
        history_after = self._native_history_marker()
        native_recorded = history_after != history_before
        if changed:
            # Store-owned snapshots are already independent arrays. Retain the
            # complete sparse request (including harmless background points)
            # to avoid several large post-commit filtered allocations for a
            # bucket fill.
            atom = (
                (rows, columns),
                erased_values,
                top_after,
                erased_values,
            )
            if not native_recorded:
                self._add_empty_native_history_atom()
            self._record(atom)
        # Grow stroke-dedup state only after aligned native/custom history is
        # durable. If this bookkeeping allocation fails, the successful
        # partial stroke remains immediately undoable.
        try:
            if compact_region:
                self.semantic_erase_compact_done = True
            elif new_linear.size:
                self.semantic_erase_seen.update(
                    int(index) for index in new_linear
                )
        except MemoryError:
            # The edit and its paired history are already durable. Stop the
            # remainder of this mouse stroke rather than risk peeling pixels
            # whose compact dedup state could not be retained.
            self.semantic_erase_seen.clear()
            self.semantic_erase_dedup_exhausted = True
            self.widget.viewer.status = (
                "This erase stroke reached the memory limit. Release the "
                "mouse and start a new stroke to continue."
            )
        if changed:
            self.widget._refresh_composite_bounds(bounds)
        return result

    def erase_visible_selection(self, selection):
        """Erase one picker-owned component without generic index expansion.

        The connected flood already guarantees unique row-major coordinates
        and binds them to a store revision.  This dedicated button path keeps
        those compact coordinates through native/custom history instead of
        repeatedly converting and sorting them like an arbitrary brush input.
        """

        if self.widget._save_worker is not None:
            raise RuntimeError(
                "Wait for the current label save to finish before deleting."
            )
        if self.block_depth or self.layer._block_history:
            raise RuntimeError(
                "Selected annotation deletion cannot join another edit."
            )
        if selection.store_revision is None:
            raise RuntimeError("The selected annotation has no store revision")

        controller = self.widget.overlap_editor
        # Picker output is frozen at construction, but retain this invariant
        # for selections supplied by older sessions or focused callers before
        # history begins sharing the arrays without copying them.
        selection.rows.setflags(write=False)
        selection.columns.setflags(write=False)
        indices = controller._unique_selection_indices(
            (selection.rows, selection.columns)
        )
        value = int(selection.value)
        projection_dtype = controller.composite.dtype
        erased_value = np.asarray(value, dtype=projection_dtype).reshape(())

        # Build and enqueue every tiny history container before the store
        # transaction. Assigning the varying top-after array into this list
        # after commit is pointer replacement and cannot allocate. This keeps
        # a deque/list allocation failure an exact no-op.
        atom = [indices, erased_value, erased_value, erased_value]
        history_item = [atom]
        native_undo_before = tuple(self.layer._undo_history)
        native_redo_before = tuple(self.layer._redo_history)
        native_staged_before = tuple(self.layer._staged_history)
        custom_undo_before = tuple(self.undo_items)
        custom_redo_before = tuple(self.redo_items)
        try:
            # The binary proxy represents whichever class is active *when
            # history is replayed*, which may differ from the picked class.
            # Never retain selected coordinates in native history: an empty
            # alignment atom lets the semantic controller update only the
            # currently relevant proxy during Undo/Redo and prevents ghost
            # memberships after a class switch.
            self._add_empty_native_history_atom()
            self.redo_items.clear()
            self.undo_items.append(history_item)
            _changed, bounds, top_after = (
                controller.erase_selected_visible_indices(
                    indices,
                    value,
                    expected_revision=int(selection.store_revision),
                )
            )
            atom[2] = top_after
        except BaseException:
            # The controller/store restore proxy, memberships, projection,
            # generation, and revision on mutation failures. Restore both
            # paired history sides here, including any maxlen eviction.
            self.layer._undo_history.clear()
            self.layer._undo_history.extend(native_undo_before)
            self.layer._redo_history.clear()
            self.layer._redo_history.extend(native_redo_before)
            self.layer._staged_history[:] = native_staged_before
            self.undo_items.clear()
            self.undo_items.extend(custom_undo_before)
            self.redo_items.clear()
            self.redo_items.extend(custom_redo_before)
            raise

        try:
            self.widget._refresh_composite_bounds(bounds)
        except Exception as error:
            # Memberships, projection, proxy, and paired history have already
            # committed. A display upload failure must not be reported as a
            # failed Delete or leave stale destructive controls armed.
            return error
        return None

    def _record(self, atom) -> None:
        self.redo_items.clear()
        if self.layer._block_history:
            self.staged.append(atom)
        else:
            self.undo_items.append([atom])

    def _move_history_bytes(self) -> int:
        return sum(
            atom.nbytes
            for queue in (self.undo_items, self.redo_items)
            for history in queue
            for atom in history
            if isinstance(atom, OverlapDelta)
        )

    def move_visible_selection(self, selection, moved):
        """Apply one planned sparse translation as a paired Undo item."""

        if self.widget._save_worker is not None:
            raise RuntimeError(
                "Wait for the current label save to finish before moving."
            )
        controller = self.widget.overlap_editor
        delta = controller.plan_object_move(
            selection.value,
            (selection.rows, selection.columns),
            (
                int(moved.rows[0] - selection.rows[0]),
                int(moved.columns[0] - selection.columns[0]),
            ),
            expected_revision=selection.store_revision,
        )
        if (
            self._move_history_bytes() + delta.nbytes
            > MAX_OBJECT_MOVE_HISTORY_BYTES
        ):
            raise MemoryError(
                "Move Undo history reached its safe memory limit. Save and "
                "reload the project before moving more large regions."
            )

        native_undo_before = tuple(self.layer._undo_history)
        native_redo_before = tuple(self.layer._redo_history)
        custom_undo_before = tuple(self.undo_items)
        custom_redo_before = tuple(self.redo_items)
        try:
            if controller.active_class == delta.value:
                self.layer._save_history(
                    (
                        delta.affected_indices,
                        delta.membership_before,
                        delta.membership_after,
                    )
                )
            else:
                self._add_empty_native_history_atom()
            self._record(delta)
            result = controller.apply_overlap_delta(
                delta,
                forward=True,
                expected_revision=selection.store_revision,
            )
        except BaseException:
            self.layer._undo_history.clear()
            self.layer._undo_history.extend(native_undo_before)
            self.layer._redo_history.clear()
            self.layer._redo_history.extend(native_redo_before)
            self.undo_items.clear()
            self.undo_items.extend(custom_undo_before)
            self.redo_items.clear()
            self.redo_items.extend(custom_redo_before)
            raise

        # Source and destination may be far apart. Two bounded uploads avoid
        # turning a small translation into a giant union-rectangle refresh.
        refresh_error = None
        for bounds in (result.source_bounds, result.destination_bounds):
            try:
                self.widget._refresh_composite_bounds(bounds)
            except Exception as error:
                # The transactional store/proxy/history move has committed.
                # A display upload failure cannot turn that success into a
                # reported data failure or roll back only one representation.
                if refresh_error is None:
                    refresh_error = error
        return delta, refresh_error

    @contextmanager
    def block_history(self):
        if self.block_depth == 0:
            self.semantic_erase_seen.clear()
            self.semantic_erase_compact_done = False
            self.semantic_erase_dedup_exhausted = False
        self.block_depth += 1
        completed = False
        try:
            with self.native_block_history():
                yield
            completed = True
        finally:
            self.block_depth -= 1
            if self.block_depth == 0:
                if not completed and self.layer._staged_history:
                    # Napari intentionally commits staged atoms only on a
                    # normal context exit. If a later callback fails, keep the
                    # earlier successful native and semantic atoms aligned as
                    # one immediately undoable partial stroke before the
                    # original exception propagates.
                    native_staged = self.layer._staged_history
                    self.layer._staged_history = []
                    self.layer._append_to_undo_history(native_staged)
                if self.staged:
                    self.undo_items.append(self.staged)
                self.staged = []
                self.semantic_erase_seen.clear()
                self.semantic_erase_compact_done = False
                self.semantic_erase_dedup_exhausted = False

    def undo(self):
        if self.widget._save_worker is not None:
            self.widget.viewer.status = (
                "Wait for the current label save to finish before undoing."
            )
            return None
        if not self.layer._undo_history:
            return None
        if self.widget._annotation_selection is not None:
            self.widget._clear_annotation_selection()
        history = self.undo_items[-1] if self.undo_items else None
        if history is None:
            return self.native_undo()
        snapshot = self._snapshot_native_history_call(
            self.layer._undo_history[-1]
        )
        if (
            self.redo_items.maxlen is not None
            and len(self.redo_items) >= self.redo_items.maxlen
        ):
            raise RuntimeError("Paired Redo history is unexpectedly full")
        try:
            self.redo_items.append(history)
        except BaseException:
            if self.redo_items and self.redo_items[-1] is history:
                self.redo_items.pop()
            raise
        try:
            result = self.native_undo()
            self.widget._restore_overlap_history(history, undoing=True)
        except BaseException:
            self.redo_items.pop()
            self._restore_native_history_call(snapshot)
            raise
        self.undo_items.pop()
        return result

    def redo(self):
        if self.widget._save_worker is not None:
            self.widget.viewer.status = (
                "Wait for the current label save to finish before redoing."
            )
            return None
        if not self.layer._redo_history:
            return None
        if self.widget._annotation_selection is not None:
            self.widget._clear_annotation_selection()
        history = self.redo_items[-1] if self.redo_items else None
        if history is None:
            return self.native_redo()
        snapshot = self._snapshot_native_history_call(
            self.layer._redo_history[-1]
        )
        if (
            self.undo_items.maxlen is not None
            and len(self.undo_items) >= self.undo_items.maxlen
        ):
            raise RuntimeError("Paired Undo history is unexpectedly full")
        try:
            self.undo_items.append(history)
        except BaseException:
            if self.undo_items and self.undo_items[-1] is history:
                self.undo_items.pop()
            raise
        try:
            result = self.native_redo()
            self.widget._restore_overlap_history(history, undoing=False)
        except BaseException:
            self.undo_items.pop()
            self._restore_native_history_call(snapshot)
            raise
        self.redo_items.pop()
        return result

    def _snapshot_native_history_call(self, native_item):
        """Capture sparse binary data and queue references before Undo/Redo."""

        data = [
            (indices, np.array(self.layer.data[indices], copy=True))
            for indices, _before, _after in native_item
        ]
        return (
            data,
            tuple(self.layer._undo_history),
            tuple(self.layer._redo_history),
            tuple(self.layer._staged_history),
            self.layer._updated_slice,
        )

    def _restore_native_history_call(self, snapshot) -> None:
        """Roll back a failed paired semantic history transition."""

        data, undo, redo, staged, updated_slice = snapshot
        for indices, values in data:
            self.layer.data[indices] = values
        self.layer._undo_history.clear()
        self.layer._undo_history.extend(undo)
        self.layer._redo_history.clear()
        self.layer._redo_history.extend(redo)
        self.layer._staged_history[:] = staged
        try:
            self.layer.refresh()
        except Exception:
            pass
        finally:
            self.layer._updated_slice = updated_slice


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
        # ``labels_path`` is the currently adopted save target.  Keep the
        # successfully loaded label input separately so editing the save-path
        # draft never changes what the Load controls refer to.
        self._loaded_labels_path: Optional[Path] = None
        self.mapping_path: Optional[Path] = None
        self._labels_output_dtype: Optional[np.dtype] = None
        self._labels_destination_identity: Optional[FileIdentity] = None
        self._mapping_destination_identity: Optional[FileIdentity] = None
        self._class_config_pending_save = False
        self._pending_deleted_value: Optional[int] = None
        self._pending_deleted_replacement: Optional[int] = None
        self._changing_selected_label = False
        self._undo_in_progress = False
        self._undo_feedback_token = 0
        self._save_worker = None
        self._active_save_path: Optional[Path] = None
        self._active_save_layer = None
        self._active_save_layer_mode = None
        self._active_save_adopt_destination = False
        self._active_save_update_class_config = False
        self._active_save_output_dtype: Optional[np.dtype] = None
        self._browse_buttons = []
        self._syncing_overlap_layer = False
        self._annotation_opacity = DEFAULT_ANNOTATION_OPACITY
        self._native_labels_controls = None
        self._native_opacity_slider = None
        self._native_semantic_label_control = None
        self._native_semantic_label_spinbox = None
        self._annotation_selection: Optional[
            AnnotationObjectSelection
        ] = None
        self._selection_layer = None
        self._selection_actions_dock = None
        self._selection_actions_owner_dock = None

        self.class_button_layout = None

        self.class_map: Dict[int, str] = {}
        self.class_colors: Dict[int, str] = {}

        self.labels_layer = None
        self.composite_layer = None
        self.overlap_store: Optional[OverlapStore] = None
        self.overlap_editor: Optional[OverlapEditorController] = None

        self._build_ui()
        self._schedule_selection_actions_dock_install()
        self._bind_hotkeys()
        # napari recreates every layer's polygon overlay when layers are added
        # or removed. Reinstall our preview callback after that core rebuild.
        self.viewer.layers.events.inserted.connect(
            self._restore_fast_rendering_after_layer_change
        )
        self.viewer.layers.events.removed.connect(
            self._restore_fast_rendering_after_layer_change
        )
        # Qt layer controls can be registered after the layer-model event or
        # recreated when layer selection changes. Retry briefly on each active
        # layer transition so the semantic class/opacity/brush bridges do not
        # depend on one event-loop turn during initial load.
        self.viewer.layers.selection.events.active.connect(
            self._schedule_native_labels_controls_install
        )
        # Shapes can reset ``editable`` while napari changes displayed
        # dimensions.  The selection outline is a read-only preview and must
        # never become an annotation editing target.
        self.viewer.dims.events.ndisplay.connect(
            self._lock_selection_preview_layer
        )

    def _restore_fast_rendering_after_layer_change(self, event=None):
        removed = getattr(event, "value", None)
        removed_selection = (
            removed is self._selection_layer
            and not self._selection_layer_is_present()
        )
        if removed_selection:
            self._selection_layer = None
            self._annotation_selection = None
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
        if removed_selection:
            self.viewer.layers.selection.active = layer
        self._schedule_native_labels_controls_install()
        self._update_project_controls()

    def _schedule_native_labels_controls_install(self, event=None) -> None:
        """Retry adaptation while napari finishes creating layer controls."""

        del event
        for delay_ms in (0, 50, 250):
            QTimer.singleShot(delay_ms, self._install_native_labels_controls)

    def _lock_selection_preview_layer(self, event=None) -> None:
        """Keep the non-authoritative outline preview read-only."""

        del event
        if int(self.viewer.dims.ndisplay) != 2:
            if self._annotation_selection is not None:
                self._clear_annotation_selection()
            return
        if not self._selection_layer_is_present():
            return
        try:
            self._selection_layer.editable = False
        except (AttributeError, RuntimeError):
            return

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.image_line = QLineEdit()
        self.label_line = QLineEdit()
        self.mapping_line = QLineEdit()

        layout.addWidget(QLabel("Histology image"))
        layout.addLayout(self._file_row(self.image_line, self._choose_image))

        layout.addWidget(QLabel("Label image to load"))
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

        # Keep destructive selection actions in their own compact widget.  A
        # real napari Viewer docks it beneath the layer list on the left;
        # ViewerModel/headless callers still own the same buttons without
        # reaching into napari's Qt implementation details.
        self.selection_actions_widget = QWidget(self)
        self.selection_actions_widget.setObjectName(
            "histologySelectedRegionActions"
        )
        self.selection_actions_widget.setSizePolicy(
            QSizePolicy.MinimumExpanding,
            QSizePolicy.Fixed,
        )
        selection_actions = QHBoxLayout(self.selection_actions_widget)
        selection_actions.setContentsMargins(6, 4, 6, 4)
        selection_actions.setSpacing(6)
        self.delete_region_btn = QPushButton("Delete…")
        self.delete_region_btn.setEnabled(False)
        self.delete_region_btn.setToolTip(
            "Confirm deletion of this visible region; hidden overlaps remain"
        )
        self.delete_region_btn.clicked.connect(
            self._delete_selected_annotation
        )
        selection_actions.addWidget(self.delete_region_btn)
        self.clear_region_btn = QPushButton("Clear")
        self.clear_region_btn.setEnabled(False)
        self.clear_region_btn.setToolTip(
            "Clear the current region selection without changing annotations"
        )
        self.clear_region_btn.clicked.connect(
            self._clear_annotation_selection
        )
        selection_actions.addWidget(self.clear_region_btn)

        self.save_destination_caption = QLabel("Save destination (editable)")
        layout.addWidget(self.save_destination_caption)
        self.save_destination_line = QLineEdit()
        self.save_destination_line.setEnabled(False)
        self.save_destination_line.setPlaceholderText("No label file loaded")
        self.save_destination_line.setToolTip(
            "Edit this path, then press Save to use guarded Save As. Existing "
            "files require confirmation and are verified before replacement."
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
            self.save_destination_line,
        ):
            line_edit.textChanged.connect(self._update_project_controls)
        self._update_project_controls()

    def _viewer_window(self):
        """Return napari's public Window facade when one is available."""

        window = getattr(self.viewer, "window", None)
        if window is None:
            return None
        if not callable(getattr(window, "add_dock_widget", None)):
            return None
        if not callable(getattr(window, "remove_dock_widget", None)):
            return None
        return window

    def _schedule_selection_actions_dock_install(self) -> None:
        """Attach the tiny selection-action panel after our main dock exists."""

        if self._viewer_window() is not None:
            QTimer.singleShot(0, self._install_selection_actions_dock)

    def _owning_dock_widget(self):
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QDockWidget):
                return parent
            parent = parent.parentWidget()
        return None

    def _install_selection_actions_dock(self) -> None:
        """Place Delete/Clear in one compact, plugin-owned left dock."""

        window = self._viewer_window()
        if window is None:
            return

        # Reopening the editor must never accumulate orphaned copies.  Use
        # napari's public mapping/removal APIs instead of inspecting private
        # dock registries or modifying the built-in layer-list widget.
        try:
            existing = window.dock_widgets.get(SELECTION_ACTIONS_DOCK_NAME)
        except (AttributeError, RuntimeError):
            existing = None
        if existing is self.selection_actions_widget:
            return
        if existing is not None:
            try:
                window.remove_dock_widget(existing)
            except (AttributeError, LookupError, RuntimeError):
                pass

        owner_dock = self._owning_dock_widget()
        try:
            dock = window.add_dock_widget(
                self.selection_actions_widget,
                name=SELECTION_ACTIONS_DOCK_NAME,
                area="left",
                allowed_areas=["left"],
                add_vertical_stretch=False,
            )
        except (AttributeError, RuntimeError, ValueError):
            self.selection_actions_widget.setParent(self)
            return

        # QMainWindow otherwise gives each newly stacked left dock a generous
        # share of the column.  Cap this one to its title and one button row so
        # napari's layer controls/list retain the available height.
        compact_height = max(
            64,
            int(self.selection_actions_widget.sizeHint().height()) + 34,
        )
        # napari calls QMainWindow.resizeDocks() while adding this dock.  A
        # smaller minimum lets that initial resize keep only the custom title
        # bar and clip the button row, notably with the Cocoa/Retina style.
        # Pin both bounds to the compact hint so the layer list loses no more
        # room than intended while the complete controls remain visible.
        dock.setMinimumHeight(compact_height)
        dock.setMaximumHeight(compact_height)
        dock.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self._selection_actions_dock = dock
        dock.destroyed.connect(self._on_selection_actions_dock_destroyed)

        self._selection_actions_owner_dock = owner_dock
        if owner_dock is not None and owner_dock is not dock:
            owner_dock.destroyed.connect(
                self._remove_selection_actions_dock
            )

    def _on_selection_actions_dock_destroyed(self, *args) -> None:
        """Keep the button widget owned if its small dock is closed."""

        del args
        self._selection_actions_dock = None
        try:
            if self.selection_actions_widget.parentWidget() is None:
                self.selection_actions_widget.setParent(self)
        except RuntimeError:
            pass

    def _remove_selection_actions_dock(self, *args) -> None:
        """Remove the companion dock when the main editor dock is closed."""

        del args
        self._selection_actions_owner_dock = None
        self._selection_actions_dock = None
        window = self._viewer_window()
        if window is not None:
            try:
                existing = window.dock_widgets.get(
                    SELECTION_ACTIONS_DOCK_NAME
                )
            except (AttributeError, RuntimeError):
                existing = None
            if existing is self.selection_actions_widget:
                try:
                    window.remove_dock_widget(existing)
                except (AttributeError, LookupError, RuntimeError):
                    pass
        try:
            if self.selection_actions_widget.parentWidget() is None:
                self.selection_actions_widget.setParent(self)
        except RuntimeError:
            pass

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
            or self._loaded_labels_path is None
            or self.mapping_path is None
        ):
            return False
        return (
            self.image_line.text().strip() == str(self.image_path)
            and self.label_line.text().strip()
            == str(self._loaded_labels_path)
            and self.mapping_line.text().strip() == str(self.mapping_path)
        )

    def _save_destination_draft_changed(self) -> bool:
        """Return whether the editable field requests guarded Save As."""
        return bool(
            self.labels_path is not None
            and self.save_destination_line.text().strip()
            != str(self.labels_path)
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
        self.save_destination_line.setEnabled(active_layer and not busy)

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
        typed_save_as = self._save_destination_draft_changed()
        rescue_required = not destination_available or (
            self._class_config_pending_save and not mapping_available
        )
        if active_layer and typed_save_as:
            self.save_btn.setText("Save As [s]")
        elif active_layer and rescue_required:
            self.save_btn.setText("Save As… [s]")
        elif self._class_config_pending_save:
            self.save_btn.setText("Save class deletion [s]")
        else:
            self.save_btn.setText("Save [s]")
        self.undo_btn.setEnabled(
            active_layer and not busy and not self._undo_in_progress
        )
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
        selection_ready = (
            active_layer
            and int(self.viewer.dims.ndisplay) == 2
            and self._annotation_selection is not None
            and self._selection_layer_is_present()
            and bool(self._selection_layer.visible)
            and not busy
        )
        self.delete_region_btn.setEnabled(selection_ready)
        self.clear_region_btn.setEnabled(selection_ready)

        self._sync_native_semantic_label_selector()

        if self.labels_path is None:
            self.save_destination_line.clear()
            self.save_destination_caption.setText("Save destination (editable)")
            self.save_destination_line.setToolTip(
                "Load a project before saving labels."
            )
            return

        destination = str(self.labels_path)
        self.save_destination_line.setToolTip(
            "Current adopted target: "
            f"{destination}\nEdit the field and press Save for guarded Save As."
        )
        if busy:
            self.save_destination_caption.setText("Saving labels to")
        elif not active_layer:
            self.save_destination_caption.setText(
                "Save disabled — loaded Labels layer was removed"
            )
        elif typed_save_as:
            self.save_destination_caption.setText(
                "New save destination — press Save to adopt after success"
            )
        elif not destination_available:
            self.save_destination_caption.setText(
                "Current target changed — edit path or use Save As"
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
            self.save_destination_caption.setText("Save destination (editable)")

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
        previous_annotation_selection = self._annotation_selection
        previous_selection_layer = self._selection_layer

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
                name="Histology",
                rgb=is_rgb,
                multiscale=len(image_pyramid) > 1,
                interpolation2d="linear",
            )
            composite_layer = self.viewer.add_labels(
                overlap_editor.composite,
                name="Annotations",
                opacity=self._annotation_opacity,
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
            selection_layer = self.viewer.add_shapes(
                [],
                ndim=2,
                name="Selected annotation outline (preview)",
                edge_color="#ffff00",
                edge_width=3,
                face_color="transparent",
            )
            selection_layer.editable = False
            selection_layer.visible = False
            self.viewer.layers.selection.active = labels_layer
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
            self._annotation_selection = previous_annotation_selection
            self._selection_layer = previous_selection_layer
            raise

        self.image_path = image_path
        self.labels_path = labels_path
        self._loaded_labels_path = labels_path
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
        self._annotation_selection = None
        self._selection_layer = selection_layer
        self._selection_layer.events.visible.connect(
            self._on_selection_layer_visibility_change
        )
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
        self.composite_layer.events.opacity.connect(
            self._on_composite_opacity_change
        )
        self._enable_overlap_edit_tracking()
        self._install_semantic_pick()
        self._install_semantic_tooltip()
        self._install_native_labels_controls()
        self._schedule_native_labels_controls_install()

        self.viewer.tooltip.visible = True

        self.image_line.setText(str(image_path))
        self.label_line.setText(str(labels_path))
        self.mapping_line.setText(str(mapping_path))
        self.save_destination_line.setText(str(labels_path))
        self.save_destination_line.setCursorPosition(0)
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
            return "Annotation tools — none"
        return f"Annotation tools — {value}: {class_map.get(value, value)}"

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
                "Label": ["Erase visible/top annotation", active_name],
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
            move_atoms = [
                atom for atom in history if isinstance(atom, OverlapDelta)
            ]
            if move_atoms:
                if len(history) != 1 or len(move_atoms) != 1:
                    raise RuntimeError(
                        "Move history must contain one sparse move command."
                    )
                result = self.overlap_editor.apply_overlap_delta(
                    move_atoms[0],
                    forward=not undoing,
                )
                self._refresh_composite_bounds(result.source_bounds)
                self._refresh_composite_bounds(result.destination_bounds)
                return
            semantic_erase_only = bool(history) and all(
                len(atom) == 4 for atom in history
            )
            erased_transaction = (
                self.overlap_editor.snapshot_erased_transaction(history)
                if semantic_erase_only
                else None
            )
            if not semantic_erase_only:
                self.overlap_editor.full_sync()
            atoms = reversed(history) if undoing else history
            combined = None
            applied_semantic = []
            try:
                for atom in atoms:
                    if len(atom) == 4:
                        indices, top_before, top_after, erased_values = atom
                        values = top_before if undoing else top_after
                        _changed, bounds = (
                            self.overlap_editor.restore_erased_indices(
                                indices,
                                erased_values,
                                values,
                                present=undoing,
                            )
                        )
                        applied_semantic.append(atom)
                    else:
                        indices, top_before, top_after = atom
                        values = top_before if undoing else top_after
                        _changed, bounds = (
                            self.overlap_editor.restore_projection_indices(
                                indices,
                                values,
                            )
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
            except BaseException:
                if erased_transaction is not None:
                    # Restore raw affected bytes/projection values captured
                    # before the first atom. This remains reliable even when
                    # the allocator keeps failing after a later atom.
                    self.overlap_editor.restore_erased_transaction(
                        erased_transaction
                    )
                else:
                    # Mixed legacy history is not emitted by one semantic
                    # erase stroke, but retain its prior inverse fallback.
                    for atom in reversed(applied_semantic):
                        indices, top_before, top_after, erased_values = atom
                        rollback_values = (
                            top_after if undoing else top_before
                        )
                        self.overlap_editor.restore_erased_indices(
                            indices,
                            erased_values,
                            rollback_values,
                            present=not undoing,
                        )
                raise
            try:
                self._refresh_composite_bounds(combined)
            except Exception:
                # Memberships, projection, binary proxy, and paired histories
                # have already transitioned atomically. A display upload
                # failure must not make the tracker roll native state back
                # while leaving the store on the new side of Undo/Redo; a
                # later pan/zoom or layer refresh will redraw authoritative
                # data.
                self.viewer.status = (
                    "Annotation history was applied, but the display could "
                    "not refresh. Pan or zoom to redraw."
                )
        finally:
            self._syncing_overlap_layer = False

    def _install_semantic_pick(self) -> None:
        """Make Pick select one connected visible semantic region."""
        layer = self.labels_layer
        layer._drag_modes = dict(layer._drag_modes)

        def semantic_pick(_layer, event):
            self._select_annotation_from_event(event)

        layer._napari_histo_semantic_pick = semantic_pick
        layer._drag_modes[Mode.PICK] = semantic_pick

    def _select_annotation_from_event(self, event) -> None:
        """Select and outline the visible component under one Pick click."""

        if self._save_worker is not None:
            self.viewer.status = (
                "Wait for the current label save to finish before selecting."
            )
            return
        if not self._labels_layer_is_active():
            return
        coordinate = mouse_event_to_labels_coordinate(
            self.composite_layer,
            event,
        )
        if coordinate is None:
            self._clear_annotation_selection()
            return
        coordinate = np.round(np.asarray(coordinate)).astype(np.intp)
        if coordinate.size != 2:
            self._clear_annotation_selection()
            self.viewer.status = (
                "Annotation selection is available only in 2-D."
            )
            return
        try:
            selected = select_visible_component(
                self.overlap_editor.composite,
                tuple(int(value) for value in coordinate),
                store_revision=getattr(self.overlap_store, "revision", None),
            )
        except SelectionTooLargeError as error:
            self._clear_annotation_selection()
            self.viewer.status = str(error)
            return
        except (MemoryError, TypeError, ValueError) as error:
            self._clear_annotation_selection()
            self.viewer.status = f"Could not select annotation: {error}"
            return
        if selected is None:
            self._clear_annotation_selection()
            self.viewer.status = "No annotation at that location."
            return
        try:
            self._set_annotation_selection(selected)
            # Pick retains the connected-region outline for deletion and
            # also makes the clicked semantic class the active paint class.
            # Avoid rebuilding the same binary membership plane when it is
            # already active; background clicks are handled above as no
            # selection and deliberately leave the paint class unchanged.
            if (
                self.overlap_editor.active_class != selected.value
                or int(self.labels_layer.selected_label) != 1
            ):
                self._select_label(selected.value)
        except Exception as error:
            # The user clicked a new object, so the previous sparse selection
            # must never remain armed when its replacement outline fails.
            self._annotation_selection = None
            if self._selection_layer_is_present():
                try:
                    self._selection_layer.data = []
                    self._selection_layer.editable = False
                    self._selection_layer.visible = False
                except Exception:
                    pass
            if self._labels_layer_is_active():
                self.viewer.layers.selection.active = self.labels_layer
            self._update_project_controls()
            self.viewer.status = (
                "The annotation selection could not be displayed; nothing "
                f"is selected ({error})."
            )

    def _selection_layer_is_present(self) -> bool:
        return bool(
            self._selection_layer is not None
            and any(
                candidate is self._selection_layer
                for candidate in self.viewer.layers
            )
        )

    @staticmethod
    def _closed_selection_paths(selection) -> list[np.ndarray]:
        paths = []
        for outline in selection.outlines:
            outline = np.asarray(outline, dtype=float)
            if outline.shape[0] < 2:
                continue
            if not np.array_equal(outline[0], outline[-1]):
                outline = np.concatenate([outline, outline[:1]], axis=0)
            paths.append(np.ascontiguousarray(outline, dtype=float))
        return paths

    def _set_annotation_selection(
        self,
        selection: AnnotationObjectSelection,
    ) -> None:
        """Show a lightweight Shapes outline for an exact sparse selection."""

        # Invalidate an earlier object before any fallible Shapes allocation
        # or tessellation. A failed new outline must never leave destructive
        # actions armed for the object the user previously selected.
        self._annotation_selection = None
        self._update_project_controls()
        paths = self._closed_selection_paths(selection)
        if not paths:
            raise ValueError("A selected annotation must have a visible outline")
        try:
            camera_zoom = float(self.viewer.camera.zoom)
        except (AttributeError, TypeError, ValueError):
            camera_zoom = 1.0
        edge_width = float(
            np.clip(3.0 / max(camera_zoom, 1e-6), 1.0, 256.0)
        )
        layer = self._selection_layer
        with warnings.catch_warnings():
            # NumPy 2.5 exposes a harmless warning in napari 0.6.6's Shapes
            # line triangulator. Limit suppression to our bounded preview.
            warnings.filterwarnings(
                "ignore",
                message="'where' used without 'out'.*",
                category=UserWarning,
                module=(
                    r"napari\.layers\.shapes\."
                    r"_accelerated_triangulate_python"
                ),
            )
            if not self._selection_layer_is_present():
                layer = self.viewer.add_shapes(
                    paths,
                    shape_type="path",
                    name="Selected annotation outline (preview)",
                    edge_color="#ffff00",
                    edge_width=edge_width,
                    face_color="transparent",
                )
                self._selection_layer = layer
                layer.events.visible.connect(
                    self._on_selection_layer_visibility_change
                )
            else:
                # The Shapes setter resets editable state in napari 0.6.
                layer.data = []
                layer.add(
                    paths,
                    shape_type="path",
                    edge_color="#ffff00",
                    edge_width=edge_width,
                    face_color="transparent",
                )
        layer.editable = False
        layer.visible = True
        self._annotation_selection = selection
        self.viewer.layers.selection.active = self.labels_layer
        self.viewer.status = (
            f"Selected visible region: class {selection.value}, "
            f"{selection.pixel_count:,} pixels."
        )
        self._update_project_controls()

    def _on_selection_layer_visibility_change(self, event=None) -> None:
        del event
        if (
            self._annotation_selection is not None
            and self._selection_layer_is_present()
            and not self._selection_layer.visible
        ):
            self.viewer.status = (
                "Selection outline is hidden; show it before Delete."
            )
            if self._labels_layer_is_active():
                self.viewer.layers.selection.active = self.labels_layer
        self._update_project_controls()

    def _clear_annotation_selection(self, checked=False) -> None:
        del checked
        self._annotation_selection = None
        if self._selection_layer_is_present():
            self._selection_layer.data = []
            self._selection_layer.editable = False
            self._selection_layer.visible = False
        if self._labels_layer_is_active():
            self.viewer.layers.selection.active = self.labels_layer
        self._update_project_controls()

    def _selected_annotation_is_current(self) -> bool:
        selection = self._annotation_selection
        if selection is None or not self._labels_layer_is_active():
            return False
        revision = getattr(self.overlap_store, "revision", None)
        if selection.store_revision is not None and revision is not None:
            # Store revisions change for every authoritative mutation. A
            # matching picker revision proves freshness without gathering and
            # comparing a multi-million-pixel projection twice around the
            # confirmation dialog.
            return int(selection.store_revision) == int(revision)
        return bool(
            np.all(
                self.overlap_editor.composite[
                    selection.rows,
                    selection.columns,
                ]
                == selection.value
            )
        )

    def _use_selected_annotation_class(self) -> None:
        selection = self._annotation_selection
        if selection is None:
            return
        self._select_label(selection.value)
        if not self._selected_annotation_is_current():
            self._clear_annotation_selection()

    def _delete_selected_annotation(self) -> None:
        """Stage deletion of one exact visible region; Save remains explicit."""

        if self._save_worker is not None:
            self.viewer.status = (
                "Wait for the current label save to finish before deleting."
            )
            return
        selection = self._annotation_selection
        if selection is None:
            return
        if not self._selected_annotation_is_current():
            self._clear_annotation_selection()
            self.viewer.status = (
                "The selected region changed. Pick it again before deleting."
            )
            return
        class_name = self.class_map.get(selection.value, str(selection.value))
        answer = QMessageBox.question(
            self,
            "Delete selected annotation?",
            f"Delete this visible region of class {selection.value}: "
            f"{class_name}?\n\n"
            f"Pixels: {selection.pixel_count:,}\n"
            "Hidden overlapping annotations will be revealed. Nothing is "
            "written to disk until you press Save.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            self.viewer.status = "Annotation deletion cancelled."
            return
        if (
            self._annotation_selection is not selection
            or not self._selection_layer_is_present()
            or not self._selection_layer.visible
        ):
            self.viewer.status = (
                "The selected region changed while confirming. Pick it again."
            )
            return
        if not self._selected_annotation_is_current():
            self._clear_annotation_selection()
            self.viewer.status = (
                "The selected region changed while confirming. Pick it again."
            )
            return
        # QMessageBox runs a nested event loop. A save hotkey may have started
        # while confirmation was open, so recheck before calling the tracker.
        if self._save_worker is not None:
            self.viewer.status = (
                "Wait for the current label save to finish before deleting."
            )
            return
        try:
            refresh_error = (
                self.labels_layer._napari_histo_edit_tracker
                .erase_visible_selection(selection)
            )
        except (MemoryError, RuntimeError, TypeError, ValueError) as error:
            self.viewer.status = f"Could not delete selected annotation: {error}"
            return
        self._clear_annotation_selection()
        if refresh_error is None:
            self.viewer.status = (
                "Selected annotation deleted in memory. Press Save to write it."
            )
        else:
            self.viewer.status = (
                "Selected annotation deleted in memory, but the display could "
                "not refresh. Pan or zoom to redraw, then press Save."
            )

    def _move_selected_annotation(
        self,
        row_direction: int,
        column_direction: int,
        *,
        step: int = 1,
    ) -> None:
        """Move a selection internally by an explicit integer step."""

        # The transactional store/history backend is installed below; keep
        # this UI handler explicit so unsupported states fail without edits.
        selection = self._annotation_selection
        if selection is None:
            return
        step = int(step)
        if step < 1:
            raise ValueError("The annotation move step must be at least one")
        row_delta = int(row_direction) * step
        column_delta = int(column_direction) * step
        try:
            moved = selection.translated(
                row_delta,
                column_delta,
                self.overlap_editor.shape,
            )
        except ValueError as error:
            self.viewer.status = str(error)
            return
        apply_move = getattr(
            self.labels_layer._napari_histo_edit_tracker,
            "move_visible_selection",
            None,
        )
        if apply_move is None:
            self.viewer.status = "Annotation movement is not available."
            return
        try:
            _delta, refresh_error = apply_move(selection, moved)
        except (MemoryError, RuntimeError, TypeError, ValueError) as error:
            self.viewer.status = f"Could not move selected annotation: {error}"
            return
        revision = getattr(self.overlap_store, "revision", None)
        moved = AnnotationObjectSelection(
            value=moved.value,
            rows=moved.rows,
            columns=moved.columns,
            outlines=moved.outlines,
            simplified_preview=moved.simplified_preview,
            store_revision=revision,
        )
        # The old coordinates became stale as soon as the transaction
        # committed. If preview tessellation fails, retain the data move and
        # disable destructive selection actions until the user picks again.
        self._annotation_selection = None
        try:
            self._set_annotation_selection(moved)
        except Exception as error:
            try:
                self._clear_annotation_selection()
            except Exception:
                self._annotation_selection = None
                self._update_project_controls()
            self.viewer.status = (
                "Annotation moved in memory, but its outline could not be "
                f"redrawn ({error}). Pan or zoom, then Pick it again."
            )
            return
        if refresh_error is not None:
            self.viewer.status = (
                "Annotation moved in memory, but the display could not "
                "refresh. Pan or zoom to redraw, then press Save when ready."
            )
            return
        self.viewer.status = (
            f"Moved selected annotation by ({row_delta}, {column_delta}) "
            "pixels in memory. Press Save to write it."
        )

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

    def _find_native_labels_controls(self):
        """Return napari's controls for the transparent edit layer, if ready."""

        if self.labels_layer is None:
            return None
        try:
            window = getattr(self.viewer, "window", None)
            qt_viewer = getattr(window, "_qt_viewer", None)
            container = getattr(qt_viewer, "controls", None)
            return getattr(container, "widgets", {}).get(self.labels_layer)
        except (AttributeError, RuntimeError):
            return None

    def _install_native_labels_controls(self, controls=None) -> bool:
        """Adapt napari's original Labels sliders to the overlap editor.

        Presentation-only brush/eraser adaptation leaves napari's actions and
        ``Labels.brush_size`` connection intact. The native opacity slider is
        redirected straight to the visible semantic composite, and the label
        field is decoupled to show semantic IDs while the edit proxy stays
        binary.
        """

        layer = self.labels_layer
        if layer is None or not any(
            candidate is layer for candidate in self.viewer.layers
        ):
            return False
        if controls is None:
            controls = self._find_native_labels_controls()
        if controls is None or getattr(controls, "layer", None) is not layer:
            return False
        if not adapt_native_labels_tool_controls(
            controls,
            maximum_brush_size=NATIVE_BRUSH_SIZE_MAX,
        ):
            return False
        try:
            opacity_control = controls._opacity_blending_controls
            opacity_slider = opacity_control.opacity_slider
            opacity_tooltip = (
                "Adjust the visible Annotations layer. The Annotation tools "
                "layer remains transparent for painting."
            )
            opacity_slider.setToolTip(opacity_tooltip)
            opacity_control.opacity_label.setToolTip(opacity_tooltip)

            label_control = controls._label_control
            semantic_spinbox = label_control.selection_spinbox
            semantic_tooltip = (
                "Active annotation class. Type a mapped class ID or use "
                "the arrows to move between available classes; 0 selects "
                "the semantic eraser."
            )
            semantic_spinbox.setToolTip(semantic_tooltip)
            label_control.label_color_label.setToolTip(semantic_tooltip)
            label_control.colorbox.setToolTip(
                "Color of the active semantic annotation class"
            )
        except (AttributeError, RuntimeError):
            return False

        already_bridged = self._native_opacity_slider is opacity_slider
        if not already_bridged:
            previous_slider = self._native_opacity_slider
            if previous_slider is not None:
                try:
                    previous_slider.valueChanged.disconnect(
                        self._on_native_opacity_slider_change
                    )
                except (TypeError, RuntimeError):
                    pass

            # napari connects this signal to ``labels_layer.opacity`` through
            # an anonymous closure. Disconnect it before installing the
            # semantic bridge; routing through the edit proxy would update its
            # thumbnail and VisPy node twice for every slider step.
            try:
                opacity_slider.valueChanged.disconnect()
            except (TypeError, RuntimeError):
                pass
            for callback in tuple(getattr(opacity_control, "_callbacks", ())):
                try:
                    layer.events.opacity.disconnect(callback)
                except (TypeError, ValueError, RuntimeError):
                    pass
            opacity_slider.valueChanged.connect(
                self._on_native_opacity_slider_change
            )

        semantic_already_bridged = (
            self._native_semantic_label_spinbox is semantic_spinbox
        )
        if not semantic_already_bridged:
            previous_spinbox = self._native_semantic_label_spinbox
            if previous_spinbox is not None:
                try:
                    previous_spinbox.valueChanged.disconnect(
                        self._on_native_semantic_label_change
                    )
                except (TypeError, RuntimeError):
                    pass

            # QtLabelControl normally mirrors Labels.selected_label in both
            # directions and resets the range from the layer's binary dtype.
            # This edit layer must remain encoded as 0/1, while the user-facing
            # selector displays real semantic class IDs. Keep the colorbox's
            # own layer-event connections: its 0/1 colormap is deliberately
            # recolored to the current semantic class.
            try:
                semantic_spinbox.valueChanged.disconnect(
                    label_control.change_selection
                )
            except (TypeError, RuntimeError):
                pass
            for callback in tuple(getattr(label_control, "_callbacks", ())):
                try:
                    layer.events.selected_label.disconnect(callback)
                except (TypeError, ValueError, RuntimeError):
                    pass
            try:
                layer.events.data.disconnect(label_control._on_data_change)
            except (TypeError, ValueError, RuntimeError):
                pass
            semantic_spinbox.valueChanged.connect(
                self._on_native_semantic_label_change
            )

        self._native_labels_controls = controls
        self._native_opacity_slider = opacity_slider
        self._native_semantic_label_control = label_control
        self._native_semantic_label_spinbox = semantic_spinbox
        self._sync_native_opacity_slider()
        self._sync_native_semantic_label_selector()
        return True

    def _on_native_opacity_slider_change(self, value: float) -> None:
        """Apply napari's native opacity control to visible annotations."""

        self._set_annotation_opacity(value)

    def _sync_native_opacity_slider(self) -> None:
        """Show semantic opacity in the edit layer's native opacity slider."""

        slider = self._native_opacity_slider
        if slider is None:
            return
        try:
            previous = slider.blockSignals(True)
            try:
                slider.setValue(self._annotation_opacity)
            finally:
                slider.blockSignals(previous)
        except RuntimeError:
            self._native_labels_controls = None
            self._native_opacity_slider = None

    def _semantic_label_selector_value(self) -> int:
        """Return the semantic class represented by the binary tool state."""

        if not self._labels_layer_is_active():
            return 0
        if int(self.labels_layer.selected_label) == 0:
            return 0
        return int(self.overlap_editor.active_class or 0)

    def _sync_native_semantic_label_selector(self) -> None:
        """Show semantic class IDs without changing the binary edit proxy."""

        spinbox = self._native_semantic_label_spinbox
        if spinbox is None:
            return
        available = [
            int(value) for value in self.class_map if int(value) > 0
        ]
        maximum = max(available, default=1)
        value = self._semantic_label_selector_value()
        maximum = max(maximum, value, 1)
        try:
            previous = spinbox.blockSignals(True)
            try:
                spinbox.setRange(0, maximum)
                spinbox.setValue(value)
                spinbox.setEnabled(self._labels_layer_is_active())
            finally:
                spinbox.blockSignals(previous)
        except RuntimeError:
            self._native_semantic_label_control = None
            self._native_semantic_label_spinbox = None

    def _on_native_semantic_label_change(self, requested: int) -> None:
        """Select a mapped semantic class from napari's native label field."""

        requested = int(requested)
        current = self._semantic_label_selector_value()
        available = sorted(
            int(value) for value in self.class_map if int(value) > 0
        )
        target = requested
        if requested not in self.class_map:
            # A one-step request comes from the native +/- buttons or wheel.
            # Move to the adjacent mapped class so sparse IDs such as 1, 23,
            # 300 remain practical to navigate. Arbitrary unknown typed IDs
            # are rejected and the selector is restored exactly.
            if requested == current + 1:
                target = next(
                    (value for value in available if value > current),
                    current,
                )
            elif requested == current - 1:
                target = next(
                    (
                        value
                        for value in reversed(available)
                        if value < current
                    ),
                    0,
                )
            else:
                self.viewer.status = (
                    f"Class {requested} is not available. Choose a mapped "
                    "class ID."
                )
                self._sync_native_semantic_label_selector()
                return

        self._select_label(target)
        self._sync_native_semantic_label_selector()

    def _set_annotation_opacity(self, value: float) -> None:
        """Set visible semantic opacity without exposing the binary proxy."""

        value = float(np.clip(value, 0.0, 1.0))
        self._annotation_opacity = value
        if self.composite_layer is not None and any(
            layer is self.composite_layer for layer in self.viewer.layers
        ):
            self.composite_layer.opacity = value
        self._sync_native_opacity_slider()

    def _on_composite_opacity_change(self, event=None) -> None:
        del event
        if self.composite_layer is None:
            return
        self._annotation_opacity = float(self.composite_layer.opacity)
        self._sync_native_opacity_slider()

    def _on_active_layer_opacity_change(self, event=None) -> None:
        del event
        if self.labels_layer is None or self.labels_layer.opacity == 0:
            return
        # The binary layer is an invisible tool proxy, never a display layer.
        # Guard programmatic writes and napari fallbacks without forwarding
        # them through the semantic opacity control.
        self.labels_layer.opacity = 0.0
        self._sync_native_opacity_slider()
        self.viewer.status = (
            "Annotation tools stay transparent. Use napari's native opacity "
            "slider to adjust the visible Annotations layer."
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
        bg_btn.setToolTip(
            "Erase the visible/top annotation and reveal underlying overlaps"
        )

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
            self.viewer.layers.selection.active = self.labels_layer
            self.viewer.status = (
                "Erase visible/top annotations; underlying overlaps will be "
                "revealed."
            )
            self._update_project_controls()
            return
        if value not in self.class_map or value not in self.overlap_store.class_values:
            self.viewer.status = f"Class {value} is not available."
            return

        class_changed = int(self.overlap_editor.active_class) != value
        # Flush any edit whose partial event was suppressed before replacing
        # the binary working mask with a *different* membership plane. Merely
        # returning from semantic Erase (selected label 0) to the already
        # active paint class must not rebuild that plane or erase Undo history.
        missed_changes = (
            self.overlap_editor.full_sync() if class_changed else 0
        )
        self._syncing_overlap_layer = True
        try:
            if class_changed:
                self.overlap_editor.select_class(value)
                # Native history stores changes to one binary class plane.
                # It cannot safely cross a genuine plane switch.
                self._limit_undo_history(self.labels_layer)
            self.labels_layer.selected_label = 1
            if missed_changes:
                self.composite_layer.refresh()
            if class_changed:
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
            self._clear_annotation_selection()
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
            self._clear_annotation_selection()
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
        if self._undo_in_progress:
            self.viewer.status = "Undo is already in progress."
            return
        if self._save_worker is not None:
            self.viewer.status = "Wait for the current label save to finish."
            return
        self._undo_in_progress = True
        self._undo_feedback_token += 1
        self.undo_btn.setText("Undoing…")
        self.undo_btn.setEnabled(False)
        self.viewer.status = "Undoing the last annotation change…"
        # repaint() is synchronous and does not dispatch input events, so the
        # user sees progress without opening a re-entrant Undo opportunity.
        self.undo_btn.repaint()
        try:
            tracker = getattr(
                self.labels_layer,
                "_napari_histo_edit_tracker",
                None,
            )
            tracker_has_paired_history = bool(
                tracker is not None and tracker.undo_items
            )
            if (
                not self._labels_layer_is_active()
                or not self._undo_labels_layer(self.labels_layer)
            ):
                self._show_undo_feedback(
                    "Nothing to undo",
                    "Nothing to undo.",
                )
                return

            # A paired tracker history already restored the packed membership
            # and sparse top projection inside ``labels_layer.undo()``.
            # Scanning the entire slide here would turn one-pixel Undo into an
            # O(slide) operation.
            if tracker_has_paired_history:
                self._show_undo_feedback(
                    "Undo complete ✓",
                    "Undo complete.",
                )
                return
            try:
                missed_changes = self.overlap_editor.full_sync()
                if missed_changes:
                    self.composite_layer.refresh()
            except (TypeError, ValueError, RuntimeError, MemoryError) as error:
                self._show_undo_feedback(
                    "Undo applied ⚠",
                    f"Undo display changed but sync failed: {error}",
                )
                return
            self._show_undo_feedback(
                "Undo complete ✓",
                "Undo complete.",
            )
        except Exception as error:
            # Paired history restores its native/custom queues and sparse data
            # transactionally before propagating a failure. Keep that safe
            # failure visible instead of letting a Qt button callback vanish
            # into a terminal traceback.
            self._show_undo_feedback(
                "Undo failed",
                f"Undo failed safely: {error}",
            )
        finally:
            self._undo_in_progress = False
            self._update_project_controls()

    def _show_undo_feedback(
        self,
        button_text: str,
        status_text: str,
        *,
        reset_after_ms: int = 1800,
    ) -> None:
        """Show deterministic Undo feedback, then restore the button label."""

        self._undo_feedback_token += 1
        token = self._undo_feedback_token
        self.undo_btn.setText(button_text)
        self.viewer.status = status_text
        self.undo_btn.repaint()

        def restore_label() -> None:
            if token != self._undo_feedback_token or self._undo_in_progress:
                return
            self.undo_btn.setText("Undo [u]")
            self._update_project_controls()

        QTimer.singleShot(max(0, int(reset_after_ms)), restore_label)

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
        draft = self.save_destination_line.text().strip()
        if draft and self._save_destination_draft_changed():
            initial = Path(draft).expanduser()
        elif self.labels_path is not None:
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

        # The native save dialog confirms an existing-file replacement. The
        # identity is still captured below and rechecked by the atomic writer.
        return self._prepare_save_as_destination(
            selected,
            confirm_existing=False,
        )

    def _prepare_save_as_destination(
        self,
        selected: str,
        *,
        confirm_existing: bool,
    ) -> Optional[tuple[Path, Optional[FileIdentity], bool]]:
        """Validate one typed/dialog Save As target without adopting it."""
        selected = str(selected).strip()
        if not selected:
            raise ValueError("Enter a PNG or TIFF save destination.")

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
            and self._paths_are_same_file(destination, self.labels_path)
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
            if protected_path is not None and self._paths_are_same_file(
                destination,
                protected_path,
            ):
                raise ValueError(
                    f"Save As cannot overwrite the {description}: "
                    f"{protected_path}"
                )

        # Recheck after resolving a final symlink. A path named ``copy.tif``
        # must not smuggle an unsupported referent into the image writer.
        if destination.suffix.lower() not in {".png", ".tif", ".tiff"}:
            raise ValueError(
                "Save As requires a PNG, TIF, or TIFF label destination."
            )

        self._validate_class_ids_for_destination(
            destination,
            self.class_map,
        )
        if expected_identity is not None and confirm_existing:
            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
            yes_button = getattr(buttons, "Yes")
            no_button = getattr(buttons, "No")
            answer = QMessageBox.question(
                self,
                "Replace existing label file?",
                "The save destination already exists:\n\n"
                f"{destination}\n\nReplace this verified file with the "
                "current annotations?",
                yes_button | no_button,
                no_button,
            )
            if answer != yes_button:
                self.viewer.status = (
                    "Save As cancelled; the existing file and current "
                    "annotations were left unchanged."
                )
                return None
        return destination, expected_identity, require_absent

    @staticmethod
    def _paths_are_same_file(first: Path, second: Path) -> bool:
        """Compare path aliases conservatively without requiring existence."""
        if first == second:
            return True
        try:
            return first.samefile(second)
        except OSError:
            return False

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
        typed_save_as = self._save_destination_draft_changed()
        use_save_as = (
            force_save_as
            or typed_save_as
            or not normal_destination_available
            or not mapping_available
        )
        if use_save_as:
            try:
                if typed_save_as and not force_save_as:
                    chosen = self._prepare_save_as_destination(
                        self.save_destination_line.text(),
                        confirm_existing=True,
                    )
                else:
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
            # This is only the editable draft. The authoritative save target,
            # identity, and dtype are adopted in _on_save_complete after the
            # worker result is independently verified.
            self.save_destination_line.setText(str(destination))
            self.save_destination_line.setCursorPosition(0)
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
            self.save_destination_line.setText(str(saved_path))
            self.save_destination_line.setCursorPosition(0)

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
