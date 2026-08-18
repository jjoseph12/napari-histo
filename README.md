# Histology Label Editor for napari

A lightweight **napari** plugin for manual editing of semantic segmentation labels on histology images.

The plugin overlays an integer-valued segmentation mask on an RGB histology image and provides a streamlined interface for editing labels using napari's native annotation tools.

---

## Features

* Load an RGB histology image and corresponding integer-valued segmentation mask
* Load a CSV mapping label values to class names
* Display semantic segmentation as a colored overlay
* Hover over any area to see its label value and class name
* One-click class selection buttons (including **Background**)
* Add new classes while napari is open
* Rename classes and change their overlay colors immediately
* Native napari editing tools:

  * Paint
  * Fill (paint bucket)
  * Erase
  * Polygon
* Adjustable overlay opacity
* Keyboard shortcut for saving
* Undo support
* Memory-efficient editing of large semantic masks
* Lower-memory connected-region fills for large label images
* Local-window Bucket searches with a safe large-region fallback
* Bounding-box polygon rendering instead of full-slide temporary masks
* Cursor-aligned, high-visibility polygon preview on oversized textures
* Partial GPU brush updates even when napari downsamples the Labels texture
* Automatic multiscale display for large RGB images
* Non-blocking, atomic saves that keep the interface responsive
* Saves edits directly back to the original label image

---

## Input Files

### Histology image

* RGB image
* Any format supported by `imageio` (PNG, TIFF, etc.)
* Must have the same dimensions as the label image

### Label image

* Single-channel integer image
* Same width and height as the histology image
* Label value `0` is reserved for background

Example:

| Pixel value | Meaning    |
| ----------- | ---------- |
| 0           | Background |
| 1           | Tumor      |
| 2           | Stroma     |
| 3           | Necrosis   |

### Class mapping CSV

A CSV with a header. The first two columns contain the numeric value and class
name. An optional `color` column stores a class color as a hex value or a
standard color name.

Example:

```csv
value,class_name,color
1,Tumor,#d1495b
2,Stroma,#2a9d8f
3,Necrosis,#f4a261
4,Lymphocytes,#6957c2
```

The first column must contain the integer label values used in the label image.
Existing two-column mapping files continue to work and use the built-in color
palette.

---

# Installation

Create (or activate) a napari environment.

Example:

```bash
mamba create -n napari-env python=3.11 napari
mamba activate napari-env
```

For a new Qt 6 environment, the equivalent conda-forge command is:

```bash
conda create -n napari-histo -c conda-forge python=3.12 napari pyqt=6
conda activate napari-histo
```

QtPy is the compatibility layer used by the plugin; `pyqt=6` is the actual
Qt 6 GUI backend. Installing from conda-forge is supported and does not
require any plugin code changes.

Clone the repository:

```bash
git clone <repository_url>
cd napari-histo-label-editor
```

Install in editable mode:

```bash
python -m pip install -e .
```

Because the package is installed in editable mode, changes to the source code take effect immediately after restarting napari.

---

# Launching

Start napari:

```bash
napari
```

Open the plugin:

```
Plugins
    → Histology Label Editor
```

---

# Loading Data

Browse to:

* Histology image
* Label image
* Class mapping CSV

Then click

```
Load
```

The plugin will display:

* the histology image
* a semantic label overlay
* one shortcut button for every class
* a shortcut button for **Background**

Hover over the label overlay to see the label value and class name at the
cursor (for example, `Label: 3 — Tumor`). The overlay can remain visible while
using this readout.

---

# Editing Labels

## Selecting a class

Click one of the colored class buttons.

Example:

```
3: Tumor
```

The selected label becomes the active paint label.

To erase objects, click

```
0: Background
```

## Adding or editing a class

Click **+ Add class**, choose a numeric value, enter the class name, and pick a
color. The new class is selected immediately and works with Paint, Fill, and
Polygon without reloading the files.

To rename or recolor an existing class, select it and click **Edit selected**.
You can also right-click its class button. Existing pixels of that class change
color immediately, and hover text immediately uses the new name.

Class additions and edits are saved directly to the mapping CSV. If a new
numeric value is larger than the current label image can store, the editor
automatically promotes the mask to a safe integer dtype before editing or
saving. Paint, Fill, and Polygon can draw the selected class over existing
classes; because this is a semantic mask, each pixel retains one class value.

---

## Paint

Choose the **Paint** tool from napari's Labels toolbar.

Paint directly onto the segmentation.

---

## Fill (Paint Bucket)

Choose the **Fill** tool.

Click inside a connected region to relabel the entire connected component.

This is particularly useful when correcting an entire segmented object.

---

## Erase

Either

* choose the **Erase** tool, or
* select **Background** and use the Fill tool.

---

## Polygon

Use napari's polygon editing tools to redraw larger regions.

---

# Adjusting Overlay Visibility

The label layer opacity can be adjusted using napari's layer controls.

Reducing opacity allows the underlying histology image to remain visible while editing.

---

# Saving

Press

```
s
```

or click

```
Save
```

The edited segmentation is written back to the original label image. Saving
runs in the background, temporarily pauses editing, and atomically replaces
the old file only after the new image is complete. Wait for the status bar to
say that saving finished before closing napari.

---

# Undo

Press

```
u
```

or click

```
Undo
```

to revert the previous edit.

Undo uses napari's changed-pixel history, so each action stores only the
edited pixels instead of copying the complete label image. Large masks are
also kept in the smallest safe integer dtype while open, reducing memory use
without changing label values or the saved file dtype. The plugin retains the
20 most recent undo actions to keep memory use predictable.

Fill and polygon tools are optimized automatically when data is loaded. A
connected fill avoids napari 0.6's full-size component-label allocation, and
starts in a bounded window around the clicked region instead of allocating a
full-slide flood map for every Bucket action. Genuinely large regions retain a
safe compiled full-image fallback. Polygon drawing allocates a temporary mask
only for the polygon's bounding box. When an editable Labels layer exceeds the
GPU texture limit, brush edits update only the changed part of napari's native
downsampled texture instead of refreshing the entire layer. The polygon tool
uses a crisp outline and compact high-contrast vertices while drawing,
compensating for napari's texture scale so the preview stays under the cursor.
Moderate RGB images stay single-scale so pan and zoom do not re-upload textures.
Truly oversized images use an antialiased, contiguous multiscale pyramid for
stable colors and fast tile uploads. These changes are transparent to the normal
napari workflow.

---

# Tips

* Zoom in before editing fine structures.
* Use **Fill** for correcting entire objects.
* Use **Background + Fill** to quickly remove incorrectly labeled objects.
* Reduce overlay opacity when tracing difficult boundaries.
* Keep the Histology layer below the Labels layer for the clearest visualization.

---

# Requirements

* Python 3.10+
* napari 0.6.x
* NumPy
* pandas
* imageio
* SciPy
* scikit-image
* QtPy

---

# Future Improvements

Potential future features include:

* Connected-component merge/split operations
* Automatic conflict detection between neighboring labels
* Keyboard shortcuts for rapid class switching
* Custom color palettes
* Morphological editing tools
* On-disk chunked editing for masks larger than available system memory
* Support for loading and saving OME-TIFF segmentations

---

# License

See the `LICENSE` file included with this repository.

---

# Acknowledgements

Built using the excellent **napari** ecosystem for interactive multidimensional image visualization and annotation.
