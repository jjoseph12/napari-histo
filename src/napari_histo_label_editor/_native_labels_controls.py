"""Small, idempotent presentation tweaks for napari Labels controls.

The overlap editor deliberately keeps napari's native tool buttons and brush
slider.  This module only adapts how those existing widgets are presented; it
does not replace their signals, actions, shortcuts, or layer modes.
"""

from __future__ import annotations

from typing import Any


DEFAULT_MAX_BRUSH_SIZE = 512
BRUSH_PAGE_STEP = 25


def adapt_native_labels_tool_controls(
    controls: Any,
    *,
    maximum_brush_size: int = DEFAULT_MAX_BRUSH_SIZE,
) -> bool:
    """Expose a large brush range and show Erase with a brush icon.

    Parameters
    ----------
    controls : napari._qt.layer_controls.qt_labels_controls.QtLabelsControls
        The native controls belonging to the editor's Labels tool layer.
    maximum_brush_size : int, optional
        Minimum upper bound for the shared Paint/Erase brush diameter.

    Returns
    -------
    bool
        ``True`` when the expected napari controls were adapted.  ``False``
        lets callers safely retry after napari creates or recreates controls.

    Notes
    -----
    ``QtModeRadioButton.mode`` stores the actual layer mode.  Its dynamic
    ``mode`` Qt property is separate and is used by napari's stylesheet only
    to select an icon.  Changing that property from ``erase`` to ``paint``
    therefore gives the eraser a brush-shaped icon while retaining its Erase
    action, tooltip, shortcut, checked state, and click behavior.
    """

    try:
        maximum_brush_size = int(maximum_brush_size)
    except (TypeError, ValueError, OverflowError):
        return False
    if maximum_brush_size < 1:
        return False

    try:
        layer = controls.layer
        brush_control = controls._brush_size_slider_control
        brush_slider = brush_control.brush_size_slider
        erase_button = controls.erase_button
        layer_brush_size = max(1, int(layer.brush_size))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False

    try:
        brush_maximum = max(
            maximum_brush_size,
            int(brush_slider.maximum()),
            layer_brush_size,
        )
        brush_slider.setMinimum(1)
        brush_slider.setMaximum(brush_maximum)
        brush_slider.setSingleStep(1)
        brush_slider.setPageStep(min(BRUSH_PAGE_STEP, brush_maximum))

        # Keep the exact numeric entry unchanged and put the full range in
        # napari's original row label. SuperQt stores a range suffix but does
        # not actually paint it in this napari/SuperQt version, so relying on
        # that suffix would leave users seeing only the current value.
        brush_slider.setEdgeLabelMode(
            brush_slider.EdgeLabelMode.LabelIsValue
        )
        brush_control.brush_size_slider_label.setText(
            f"brush size 1–{brush_maximum}:"
        )
        brush_tooltip = (
            f"Paint/Erase brush diameter: 1–{brush_maximum} pixels. "
            "Drag from small to large, use arrow keys for 1-pixel changes, "
            "or click the number to type an exact size."
        )
        brush_slider.setToolTip(brush_tooltip)
        brush_control.brush_size_slider_label.setToolTip(brush_tooltip)

        if erase_button.property("mode") != "paint":
            erase_button.setProperty("mode", "paint")
            # Dynamic-property selectors are not always recalculated by Qt
            # until the widget is repolished.
            style = erase_button.style()
            style.unpolish(erase_button)
            style.polish(erase_button)
        erase_button.update()
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False

    return True
