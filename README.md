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
* Safely delete classes, with optional pixel reassignment staged until Save
* Preserve multiple class memberships at the same pixel without destructive
  overwriting
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
* Save the ordinary 2-D projection and lossless overlap data together inside
  the original PNG or TIFF—no sidecar files

---

## Input Files

### Histology image

* RGB image
* Any format supported by `imageio` (PNG, TIFF, etc.)
* Must have the same dimensions as the label image

### Label image

* PNG or TIFF single-channel integer image
* Same width and height as the histology image
* Label value `0` is reserved for background

Existing ordinary 2-D masks load normally. Once **Save** is pressed, the same
imported PNG or TIFF still contains an ordinary 2-D top-class projection, so
standard image readers can open it as before. The plugin also embeds every
class membership, the currently visible class at each pixel, and the class
metadata inside that one file. It does not create a sidecar file.

PNG label images support class IDs through `65535`. Use TIFF when class IDs
need to be larger than `65535`.

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

The selected class becomes the active annotation class.

To erase the active class membership, click

```
0: Background
```

## Adding, editing, or deleting a class

Click **+ Add class**, choose a numeric value, enter the class name, and pick a
color. The new class is selected immediately and works with Paint, Fill, and
Polygon without reloading the files.

To rename or recolor an existing class, select it and click **Edit selected**.
You can also right-click its class button. Existing pixels of that class change
color immediately, and hover text immediately uses the new name.

Class additions and edits are saved directly to the mapping CSV. Annotation
membership changes still wait for **Save**. If a new
numeric value is larger than the current label image can store, the editor
automatically promotes the mask to a safe integer dtype before editing or
saving. PNG class IDs cannot exceed `65535`; choose a TIFF label image for
larger IDs.

To remove a class, select it and click **Delete selected…**. If the class is
used in the mask, choose the class its pixels should become; **Background** is
the default. An unused class can be removed without a replacement. Background
(value `0`) is reserved and cannot be deleted. Deleting to Background removes
only that class membership and can reveal another class underneath. Reassigning
to a class transfers the deleted membership while preserving all other
memberships at those pixels.

Class deletion is irreversible and clears the current Undo history so an old
Undo operation cannot restore a removed value. The deletion and any pixel
reassignment are only staged in memory at first:
neither the label image nor the class mapping CSV changes on disk until you
press **Save**. Closing or loading another project before saving leaves both
original files unchanged. While a deletion is pending, save it before adding,
editing, or deleting another class.

---

## How overlapping annotations work

Paint, Polygon, and Fill add the active class membership without deleting any
classes already present underneath it. The active class becomes the visible
top class only at pixels touched by that edit. Repainting a class that is
already present brings it back to the top at the touched pixels without
creating a duplicate membership.

Erase removes only the active class membership. If another class is present at
the same pixel, it is revealed instead of being deleted. When several hidden
memberships remain, the revealed class follows a stable saved fallback order.
The format preserves every membership and the current visible top class, but it
does not store a complete chronological stack of every past paint operation at
each pixel.

The Pick tool uses the visible semantic projection: picking a displayed class
selects that semantic class rather than the internal binary edit value.

All Paint, Polygon, Fill, Erase, repaint, and Pick-driven annotation changes
remain in memory until **Save** is pressed.

---

## Paint

Choose the **Paint** tool from napari's Labels toolbar.

Paint the active class directly onto the segmentation. Existing memberships
under the stroke are retained.

---

## Fill (Paint Bucket)

Choose the **Fill** tool.

Click inside a connected visible region to add the active class membership to
that connected component.

This is particularly useful when correcting an entire segmented object.

---

## Erase

Either

* choose the **Erase** tool, or
* select **Background** and use the Fill tool.

Only the active class membership is removed; a hidden class underneath it is
revealed.

---

## Polygon

Use napari's polygon editing tools to add the active class over larger regions.
Classes already present in the polygon remain stored underneath it.

---

# Adjusting Overlay Visibility

Use the plugin's **Overlay opacity** slider to adjust the visible annotations.
The napari Labels controls operate on a transparent editing proxy; its opacity
is forced to zero so it cannot cover the semantic composite.

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

The edited annotations are written back to the exact PNG or TIFF selected by
the last successful **Load**. Saving runs in the background, temporarily pauses
editing, and atomically replaces the old file only after the new image is
complete. Wait for the status bar to say that saving finished before closing
napari.

The saved file contains two views of the same annotations:

* an ordinary 2-D top-class projection that `imageio`, pathology tools, and
  other generic image readers can still open; and
* private embedded data containing every overlapping membership, the current
  top class, saved fallback order, names, and colors needed for an exact plugin
  reload.

Both views stay inside the original file. No `.npz`, auxiliary mask, or other
sidecar is created. Paint, Polygon, Fill, Erase, and overlap-order changes do
not alter this file until **Save** is pressed.

Generic image editors usually preserve the visible 2-D projection but may
strip private embedded data when they rewrite a PNG or TIFF. Such a rewrite can
permanently remove hidden overlaps. Keep a backup and use this plugin to save
files that must retain lossless overlapping annotations.

Class additions, renames, and color edits update the mapping CSV immediately.
If a class deletion is pending, **Save** first commits the lossless label image
and then updates the class mapping CSV. Until **Save** is pressed, a pending
deletion changes neither file on disk.

The exact canonical file used by Save is always shown in the read-only
**Save destination (locked by Load)** field. Browsing to or typing different
project paths does not redirect a save. Until the new files pass **Load**, the
canvas and Save action still belong to the previous project, the button reads
**Save locked file**, and the exact destination remains visible. Loading is
disabled while a save is running. If the locked destination file is moved,
deleted, or replaced after loading, the editor refuses to write until it is
selected and loaded again.

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

Switching to a different semantic class clears the current Undo/Redo history
because napari's single binary editing proxy is reused for the newly selected
class.

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
* tifffile
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
