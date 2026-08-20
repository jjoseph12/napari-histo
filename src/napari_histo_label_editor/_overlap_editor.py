"""Pure NumPy runtime bridge for editing overlapping semantic labels.

``OverlapStore`` owns one boolean membership plane per semantic class.  A
napari ``Labels`` layer, however, can edit only one scalar array.  This module
bridges those models with two stable 2-D arrays:

* ``composite`` is a read-only projection of every class; and
* ``edit_mask`` is a writable binary copy of the active class plane.

The binary layer is intended to stay transparent while receiving napari's
Paint, Fill, and Polygon tools. Changes are projected into ``composite`` as
small rectangles, so class switching only unpacks one bit plane and never
reprojects a slide-sized array. Undo and redo can use :meth:`full_sync` as a
safe fallback without coupling this module to Qt or napari events.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ._overlap_store import OverlapStore


__all__ = ["OverlapEditorController"]


class OverlapEditorController:
    """Synchronize a projected label image and one active-class edit mask.

    Parameters
    ----------
    store : OverlapStore
        Membership store backing the editor.
    active_class : int, optional
        Initial non-background class to edit.  When omitted, the topmost
        class in ``store.z_order`` is selected.  A background-only store has
        no active class and receives an all-zero edit mask.
    projection_dtype : numpy dtype, optional
        Dtype for the runtime composite.  By default the store chooses the
        smallest safe integer dtype.

    Notes
    -----
    Both public arrays keep stable identities across ordinary edits and class
    switches.  That lets a UI give them directly to napari once and update
    the corresponding layers with partial refresh events.
    """

    def __init__(
        self,
        store: "OverlapStore",
        active_class: int | None = None,
        *,
        projection_dtype: npt.DTypeLike | None = None,
    ) -> None:
        self.store = store
        self._shape = self._validate_shape(store.shape)
        self._projection_dtype = (
            None
            if projection_dtype is None
            else np.dtype(projection_dtype)
        )

        if active_class is None:
            active_class = self._default_active_class()
        self._active_class = self._validate_active_class(active_class)

        self._composite = self._validate_projection(
            store.projection_view
        )

        self._edit_mask = np.zeros(self._shape, dtype=np.uint8)
        self._load_active_plane()

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial shape shared by the store and both runtime arrays."""

        return self._shape

    @property
    def active_class(self) -> int | None:
        """Currently edited non-background class, or ``None``."""

        return self._active_class

    @property
    def composite(self) -> np.ndarray:
        """Stable, read-only topmost-class projection of the complete store."""

        return self._composite

    @property
    def edit_mask(self) -> np.ndarray:
        """Stable writable ``uint8`` mask for :attr:`active_class`.

        Values are always zero or one.  A caller modifying this array directly
        must subsequently call :meth:`process_indices`,
        :meth:`process_changed_patch`, or :meth:`full_sync`.
        """

        return self._edit_mask

    def select_class(self, value: int) -> np.ndarray:
        """Switch the binary editor to ``value`` without reprojecting.

        Patch events should be processed before switching.  This method does
        not scan the previous full-resolution edit plane, keeping class
        selection from doing a redundant store update.

        Returns
        -------
        numpy.ndarray
            The stable :attr:`edit_mask` object, populated for ``value``.
        """

        selected = self._validate_active_class(value)
        if selected is None:
            raise ValueError("Background cannot be an active edit class.")
        if selected == self._active_class:
            # Re-read so a caller can deliberately discard unsynchronized
            # local edits and restore the authoritative store plane.
            self._load_active_plane()
            return self._edit_mask

        self._active_class = selected
        self._load_active_plane()
        return self._edit_mask

    def process_patch(
        self,
        patch: npt.ArrayLike,
        offset: Sequence[int],
    ) -> int:
        """Apply one updated binary rectangle to the active store plane.

        ``patch`` is also copied into :attr:`edit_mask`, making this method
        useful both for napari events and direct programmatic edits. The same
        rectangle is reprojected into :attr:`composite` after the store update.

        Returns
        -------
        int
            Number of membership pixels changed in the store.
        """

        active = self._require_active_class()
        binary_patch = self._as_binary_patch(patch)
        row_start, column_start = self._validate_region(
            offset,
            binary_patch.shape,
        )
        if binary_patch.size == 0:
            return 0

        row_stop = row_start + binary_patch.shape[0]
        column_stop = column_start + binary_patch.shape[1]
        self._edit_mask[
            row_start:row_stop,
            column_start:column_stop,
        ] = binary_patch
        changed = int(
            self.store.update_patch(
                active,
                (row_start, column_start),
                binary_patch,
            )
        )
        if changed:
            self.refresh_projection_patch(
                (row_start, column_start),
                binary_patch.shape,
            )
        return changed

    def process_changed_patch(
        self,
        offset: Sequence[int],
        shape: Sequence[int],
    ) -> int:
        """Synchronize a changed rectangle already present in ``edit_mask``.

        This form is preferable for napari ``labels_update`` events because
        their data can be texture-transformed.  The event's offset and shape
        identify the rectangle, while the raw binary array remains the source
        of truth.
        """

        patch_shape = self._validate_patch_shape(shape)
        row_start, column_start = self._validate_region(offset, patch_shape)
        row_stop = row_start + patch_shape[0]
        column_stop = column_start + patch_shape[1]
        return self.process_patch(
            self._edit_mask[
                row_start:row_stop,
                column_start:column_stop,
            ],
            (row_start, column_start),
        )

    def process_indices(self, indices: Any) -> int:
        """Synchronize the smallest rectangle containing changed indices.

        Accepted inputs are a NumPy-style ``(rows, columns)`` tuple, an
        ``(N, 2)`` coordinate array, or a 2-D boolean mask.  Values are read
        from the already-modified :attr:`edit_mask`; coordinate arrays are not
        expanded into another full-slide mask.
        """

        bounds = self._index_bounds(indices)
        if bounds is None:
            return 0
        row_start, row_stop, column_start, column_stop = bounds
        return self.process_changed_patch(
            (row_start, column_start),
            (row_stop - row_start, column_stop - column_start),
        )

    def normalize_indices(self, indices: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return validated, unique sparse row/column coordinates."""

        rows, columns = self._coordinate_arrays(indices)
        if rows.size == 0:
            return rows.astype(np.intp), columns.astype(np.intp)
        linear = (
            rows.astype(np.intp, copy=False) * self._shape[1]
            + columns.astype(np.intp, copy=False)
        )
        unique = np.unique(linear)
        return unique // self._shape[1], unique % self._shape[1]

    def raise_active_indices(
        self,
        indices: Any,
    ) -> tuple[int, tuple[int, int, int, int] | None]:
        """Raise the active membership only at explicitly touched pixels."""

        active = self._require_active_class()
        normalized = self.normalize_indices(indices)
        bounds = self._bounds_for_coordinates(*normalized)
        changed = self.store.raise_projection_indices(active, normalized)
        return int(changed), bounds

    def projection_values(self, indices: Any) -> np.ndarray:
        """Copy authoritative top values at unique sparse coordinates."""

        rows, columns = self.normalize_indices(indices)
        return np.array(self._composite[rows, columns], copy=True)

    def restore_projection_indices(
        self,
        indices: Any,
        values: npt.ArrayLike,
    ) -> tuple[int, tuple[int, int, int, int] | None]:
        """Restore validated sparse top values for exact Undo/Redo."""

        normalized = self.normalize_indices(indices)
        bounds = self._bounds_for_coordinates(*normalized)
        changed = self.store.restore_projection_indices(normalized, values)
        return int(changed), bounds

    def full_sync(self) -> int:
        """Replace the active store plane from the complete binary edit mask.

        Use this after napari Undo/Redo, which refreshes the layer without
        emitting the same changed-index event as a paint operation. A changed
        full-plane sync also refreshes the shared projection reference.
        """

        if self._active_class is None:
            changed = int(np.count_nonzero(self._edit_mask))
            if changed:
                self._edit_mask.fill(0)
                self.refresh_projection()
            return changed
        active = self._active_class
        changed, _additions, _removals = self.store.update_plane_counts(
            active,
            self._edit_mask,
        )
        changed = int(changed)
        if changed:
            self.refresh_projection()
        return changed

    def projection_patch(
        self,
        offset: Sequence[int],
        shape: Sequence[int],
    ) -> np.ndarray:
        """Return a fresh all-class composite patch."""

        patch_shape = self._validate_patch_shape(shape)
        normalized_offset = self._validate_region(offset, patch_shape)
        patch = self.store.project_patch(
            normalized_offset,
            patch_shape,
            dtype=self._composite.dtype,
            exclude=None,
        )
        patch = np.asarray(patch)
        if patch.shape != patch_shape:
            raise ValueError(
                "OverlapStore returned projection patch shape "
                f"{patch.shape}; expected {patch_shape}."
            )
        return patch

    def refresh_projection_patch(
        self,
        offset: Sequence[int],
        shape: Sequence[int],
    ) -> np.ndarray:
        """Reproject and replace one bounded rectangle in ``composite``."""

        patch_shape = self._validate_patch_shape(shape)
        row_start, column_start = self._validate_region(offset, patch_shape)
        row_stop = row_start + patch_shape[0]
        column_stop = column_start + patch_shape[1]
        return self._read_only_view(
            self._composite[
                row_start:row_stop,
                column_start:column_stop,
            ]
        )

    # Concise alias for event adapters that already use "project patch" as
    # the refresh verb.
    project_patch = refresh_projection_patch

    def refresh_projection(self) -> np.ndarray:
        """Refresh the shared projection reference after a dtype change."""

        self.refresh_projection_reference()
        return self._composite

    def refresh_projection_reference(self) -> bool:
        """Rebind to the store projection if class growth changed its dtype."""

        projection = self._validate_projection(self.store.projection_view)
        if projection is self._composite:
            return False
        if (
            projection.dtype == self._composite.dtype
            and np.shares_memory(projection, self._composite)
        ):
            return False
        self._composite = projection
        return True

    def saved_projection(
        self,
        *,
        dtype: npt.DTypeLike | None = None,
    ) -> np.ndarray:
        """Return the compatibility projection containing every class.

        This is the 2-D image to write through the legacy label-image path.
        The embedded overlap payload remains the lossless source for all
        memberships.
        """

        target_dtype = self._projection_dtype if dtype is None else np.dtype(dtype)
        projection = self.store.project(dtype=target_dtype, exclude=None)
        return self._validate_projection(projection)

    def delete_class(self, value: int, replacement: int = 0) -> int:
        """Remove a class, optionally OR-reassigning it into another class.

        Only pixels in the deleted membership plane can change the projected
        base, so a non-active deletion refreshes that plane's bounding box.
        Deleting the active class selects the replacement when possible (or
        the next topmost class) and performs a full projection refresh.

        Returns
        -------
        int
            Number of pixels that belonged to the deleted class.
        """

        deleted = self._validate_non_background_value(value, "Deleted class")
        replacement_value = int(replacement)
        if replacement_value < 0:
            raise ValueError("Replacement class cannot be negative.")
        if replacement_value == deleted:
            raise ValueError("A deleted class cannot replace itself.")
        if replacement_value and replacement_value not in self.store.class_values:
            raise ValueError(
                f"Replacement class {replacement_value} is not in the store."
            )

        removed_pixels = int(self.store.count_class(deleted))
        deleting_active = deleted == self._active_class

        self.store.remove_class(
            deleted,
            replacement=None if replacement_value == 0 else replacement_value,
        )

        if deleting_active:
            if replacement_value and replacement_value in self.store.class_values:
                self._active_class = replacement_value
            else:
                self._active_class = self._default_active_class()
            self._load_active_plane()
            self.refresh_projection()
        else:
            # OR-reassignment into the current class changes the writable
            # proxy as well as the authoritative projection.
            if replacement_value == self._active_class:
                self._load_active_plane()
            self.refresh_projection()

        return removed_pixels

    def reassign_class(self, value: int, replacement: int) -> int:
        """Alias for :meth:`delete_class` with a required replacement."""

        if int(replacement) == 0:
            raise ValueError("Reassignment requires a non-background class.")
        return self.delete_class(value, replacement)

    def _load_active_plane(self) -> None:
        if self._active_class is None:
            self._edit_mask.fill(0)
            return
        plane = np.asarray(self.store.select_plane(self._active_class))
        if plane.shape != self._shape:
            raise ValueError(
                "OverlapStore returned class plane shape "
                f"{plane.shape}; expected {self._shape}."
            )
        self._edit_mask[...] = plane != 0

    def _default_active_class(self) -> int | None:
        class_values = set(int(value) for value in self.store.class_values)
        for value in reversed(tuple(self.store.z_order)):
            value = int(value)
            if value in class_values:
                return value
        if class_values:
            return max(class_values)
        return None

    def _validate_active_class(self, value: int | None) -> int | None:
        if value is None:
            if self.store.class_values:
                raise ValueError(
                    "An active class is required when the store has classes."
                )
            return None
        selected = self._validate_non_background_value(value, "Active class")
        if selected not in self.store.class_values:
            raise ValueError(f"Class {selected} is not in the overlap store.")
        return selected

    def _require_active_class(self) -> int:
        if self._active_class is None:
            raise RuntimeError("No non-background class is available to edit.")
        return self._active_class

    @staticmethod
    def _validate_non_background_value(value: int, description: str) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{description} must be an integer class value.")
        try:
            normalized = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(
                f"{description} must be an integer class value."
            ) from error
        if normalized <= 0:
            raise ValueError(f"{description} must be greater than zero.")
        return normalized

    @staticmethod
    def _validate_shape(shape: Sequence[int]) -> tuple[int, int]:
        if len(shape) != 2:
            raise ValueError(f"Overlap store must be 2-D. Got shape {shape}.")
        result = tuple(int(length) for length in shape)
        if any(length < 0 for length in result):
            raise ValueError(f"Overlap store shape cannot be negative: {shape}.")
        return result

    def _validate_projection(self, projection: npt.ArrayLike) -> np.ndarray:
        array = np.asarray(projection)
        if array.shape != self._shape:
            raise ValueError(
                "OverlapStore returned projection shape "
                f"{array.shape}; expected {self._shape}."
            )
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(
                "OverlapStore projection must use an integer dtype. "
                f"Got {array.dtype}."
            )
        return array

    @staticmethod
    def _validate_patch_shape(shape: Sequence[int]) -> tuple[int, int]:
        if len(shape) != 2:
            raise ValueError(f"Patch must be 2-D. Got shape {shape}.")
        result = tuple(int(length) for length in shape)
        if any(length < 0 for length in result):
            raise ValueError(f"Patch shape cannot be negative: {shape}.")
        return result

    def _validate_region(
        self,
        offset: Sequence[int],
        shape: Sequence[int],
    ) -> tuple[int, int]:
        if len(offset) != 2:
            raise ValueError(f"Patch offset must have two values. Got {offset}.")
        if any(isinstance(value, (bool, np.bool_)) for value in offset):
            raise TypeError("Patch offsets must be integers.")
        row_start, column_start = (int(value) for value in offset)
        patch_height, patch_width = self._validate_patch_shape(shape)
        if row_start < 0 or column_start < 0:
            raise ValueError(f"Patch offset cannot be negative: {offset}.")
        if (
            row_start + patch_height > self._shape[0]
            or column_start + patch_width > self._shape[1]
        ):
            raise ValueError(
                f"Patch at {tuple(offset)} with shape {(patch_height, patch_width)} "
                f"exceeds store shape {self._shape}."
            )
        return row_start, column_start

    @staticmethod
    def _as_binary_patch(patch: npt.ArrayLike) -> np.ndarray:
        array = np.asarray(patch)
        if array.ndim != 2:
            raise ValueError(f"Edit patch must be 2-D. Got shape {array.shape}.")
        if array.dtype == np.bool_:
            return array.astype(np.uint8, copy=False)
        if not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        ):
            raise TypeError(
                "Edit patch must contain binary numeric values. "
                f"Got {array.dtype}."
            )
        if array.size and not np.all((array == 0) | (array == 1)):
            raise ValueError("Edit patch values must be zero or one.")
        return array.astype(np.uint8, copy=False)

    def _index_bounds(
        self,
        indices: Any,
    ) -> tuple[int, int, int, int] | None:
        rows, columns = self.normalize_indices(indices)
        return self._bounds_for_coordinates(rows, columns)

    def _coordinate_arrays(
        self,
        indices: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(indices, tuple):
            if len(indices) != 2:
                raise ValueError("Changed indices must have row and column arrays.")
            rows, columns = np.broadcast_arrays(
                np.asarray(indices[0]),
                np.asarray(indices[1]),
            )
        else:
            coordinates = np.asarray(indices)
            if coordinates.dtype == np.bool_:
                if coordinates.shape != self._shape:
                    raise ValueError(
                        "Boolean changed-index mask must match editor shape "
                        f"{self._shape}; got {coordinates.shape}."
                    )
                rows, columns = np.nonzero(coordinates)
            elif coordinates.ndim == 2 and coordinates.shape[1] == 2:
                rows, columns = coordinates[:, 0], coordinates[:, 1]
            else:
                raise ValueError(
                    "Changed indices must be a (rows, columns) tuple, an "
                    "(N, 2) array, or a 2-D boolean mask."
                )

        rows = np.asarray(rows).reshape(-1)
        columns = np.asarray(columns).reshape(-1)
        if rows.size == 0:
            return rows, columns
        if not (
            np.issubdtype(rows.dtype, np.integer)
            and np.issubdtype(columns.dtype, np.integer)
        ):
            raise TypeError("Changed indices must be integers.")
        row_min = int(rows.min())
        row_max = int(rows.max())
        column_min = int(columns.min())
        column_max = int(columns.max())
        if (
            row_min < 0
            or column_min < 0
            or row_max >= self._shape[0]
            or column_max >= self._shape[1]
        ):
            raise IndexError(
                "Changed indices exceed editor shape "
                f"{self._shape}: rows {row_min}..{row_max}, columns "
                f"{column_min}..{column_max}."
            )
        return rows, columns

    @staticmethod
    def _bounds_for_coordinates(
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> tuple[int, int, int, int] | None:
        if rows.size == 0:
            return None
        return (
            int(rows.min()),
            int(rows.max()) + 1,
            int(columns.min()),
            int(columns.max()) + 1,
        )

    @staticmethod
    def _read_only_view(array: np.ndarray) -> np.ndarray:
        view = array.view()
        view.flags.writeable = False
        return view
