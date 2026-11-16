import re
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.feature import match_template


def register_pair(a, b, exp_dx, exp_dy, radius_x, radius_y):
    """
    Offset (dx, dy) of tile b's origin relative to tile a's origin,
    searched within +/- radius_x, radius_y of the expected offset.

    The reference is the expected overlap region of `a`, shrunk by the
    search radius on each side, so the match can move in both directions
    (the window is symmetric around the expected offset).

    Returns dx, dy, correlation, at_edge. at_edge=True means the best match
    sits on the border of the search window, i.e. the true offset is
    probably outside it, so the measurement should not be trusted.
    """
    H, W = a.shape
    exp_dx, exp_dy = round(exp_dx), round(exp_dy)

    # Overlap region in a's coordinates, shrunk by the search radius
    x0 = max(0, exp_dx) + radius_x
    x1 = min(W, W + exp_dx) - radius_x
    y0 = max(0, exp_dy) + radius_y
    y1 = min(H, H + exp_dy) - radius_y
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise ValueError(
            "Search radius is too large for the tile overlap: "
            f"reference would be {x1 - x0} x {y1 - y0} px."
        )
    reference = a[y0:y1, x0:x1]

    # The same region in b's coordinates, grown by the search radius
    sx0 = x0 - exp_dx - radius_x
    sy0 = y0 - exp_dy - radius_y
    search = b[sy0:y1 - exp_dy + radius_y, sx0:x1 - exp_dx + radius_x]

    result = match_template(search, reference, pad_input=False)
    by, bx = np.unravel_index(np.argmax(result), result.shape)
    at_edge = bx in (0, result.shape[1] - 1) or by in (0, result.shape[0] - 1)

    dx = x0 - (sx0 + bx)
    dy = y0 - (sy0 + by)
    return int(dx), int(dy), float(result[by, bx]), bool(at_edge)


def consensus_step(measurements, tol):
    """
    Most common neighbour step among trusted measurements: the stage
    step that the largest number of pairs agree on (within tol px).

    Returns (step_dx, step_dy), n_agreeing, or (None, 0) if there are no
    trusted measurements.
    """
    d = np.array([(m[2], m[3]) for m in measurements if m[5]], dtype=float)
    if len(d) == 0:
        return None, 0
    counts = [(np.abs(d - p).max(axis=1) <= tol).sum() for p in d]
    best = d[int(np.argmax(counts))]
    agree = np.abs(d - best).max(axis=1) <= tol
    return tuple(d[agree].mean(axis=0)), int(agree.sum())


def downsample(image, factor, method="decimate"):
    """Reduce resolution by an integer factor along both axes.

    method="decimate": keep every factor-th pixel (fast; what you asked for)
    method="mean":     average each factor x factor block (proper binning: less noise, no aliasing)
    """
    if factor == 1:
        return image
    if method == "decimate":
        return image[::factor, ::factor]
    if method == "mean":
        h = image.shape[0] // factor * factor          # trim so the size divides evenly
        w = image.shape[1] // factor * factor
        blocks = image[:h, :w].astype(np.float32).reshape(
            h // factor, factor, w // factor, factor, *image.shape[2:]
        )
        out = blocks.mean(axis=(1, 3))
        if np.issubdtype(image.dtype, np.integer):
            out = np.rint(out)
        return out.astype(image.dtype)
    raise ValueError(f"Unknown method: {method}")


def save_tiff(path, image, pixel_size_um=None):
    """Write a TIFF, embedding the plate scale so Fiji/ImageJ shows real units."""
    kwargs = {}
    if pixel_size_um is not None:
        px_per_cm = 1e4 / pixel_size_um
        kwargs = dict(resolution=(px_per_cm, px_per_cm), resolutionunit="CENTIMETER")
    tifffile.imwrite(path, image, **kwargs)
    print(f"Saved {image.shape[1]} x {image.shape[0]} TIFF to: {path}")

def make_display_image(image):
    """Stretch to 8-bit for viewing: 0.5-99.5 percentile range mapped to 0-255."""
    if image.dtype == np.uint8:
        return image

    image_float = image.astype(np.float32)
    low, high = np.percentile(image_float, [0.5, 99.5])
    if high <= low:
        low, high = image_float.min(), image_float.max()

    image_float = np.clip((image_float - low) / (high - low), 0, 1)
    return (image_float * 255).astype(np.uint8)

def stitch_nikon_tiles(
    input_folder,
    output_file="stitched.tif",
    overlap_fraction=0.05,
    search_fraction=0.03,
    max_x=5,
    max_y=5,
    factor=1,              # downsample factor for the low-res outputs (1 = none)
    method="decimate",     # "decimate" or "mean"
    save_full_tiff=True,   # save the full-resolution stitched TIFF
    save_lowres_tiff=True, # save the downsampled TIFF (only if factor > 1)
    save_png=True,         # save a viewable 8-bit PNG of the downsampled image
    pixel_size_um=None,    # input plate scale, e.g. 3.0 -> written into the TIFF metadata
    stage_step=None,       # ((dx, dy) horizontal, (dx, dy) vertical) neighbour steps in px; None = estimate
    refine=True,           # refine each tile around the grid using image matching
    max_dev=15,            # px; matches further than this from the grid step are rejected
    prior_weight=0.1,      # pull toward the regular grid (relative to a good match = 1)
    highpass_sigma=30,     # px; removes vignetting / bright spot before matching (0 = off)
    verbose=False,         # print every pairwise measurement
):
    """
    Stitch a Nikon NIS Elements XY tile set into one large TIFF image.

    Expected filenames:
        tile_x001_y001.tif
        tile_x002_y001.tif
        ...
        tile_x005_y005.tif

    Parameters
    ----------
    input_folder : str or Path
        Folder containing the tile TIFF files.

    output_file : str or Path
        Filename for the stitched TIFF.

    overlap_fraction : float
        Expected fractional overlap between neighbouring tiles.
        For 5% overlap use 0.05.

    search_fraction : float
        Fraction of the image width or height searched across the
        overlap (perpendicular to it) when estimating the stage step.

    max_x : int
        Number of tiles in the x direction.

    max_y : int
        Number of tiles in the y direction.

    stage_step : ((float, float), (float, float)) or None
        Offset in pixels between horizontal neighbours and between
        vertical neighbours, e.g. ((5696, -86), (56, 3776)). If None,
        it is estimated from the pairs that agree with each other.

    refine : bool
        If False, tiles are placed exactly on the stage grid.
        If True, each tile may move up to ~max_dev px from the grid
        where image matching is reliable.

    How positions are found
    -----------------------
    1. Every neighbouring pair is matched within a window that is
       symmetric around the expected offset (high-pass filtered so the
       vignetting / bright spot shared by all tiles doesn't dominate).
    2. The stage is very repeatable, so the true step is the one most
       pairs agree on (the "grid prior").
    3. Positions are solved by least squares: every pair is weakly pulled
       toward the grid step, and matches within max_dev px of the grid
       refine it. Failed matches (blank background tiles, wrong peaks)
       are rejected, so those tiles simply sit on the grid.

    Returns
    -------
    stitched : numpy.ndarray
        Final stitched image.

    positions : dict
        Global position of each tile in pixels.

    """

    input_folder = Path(input_folder)
    output_file = Path(output_file)

    # ------------------------------------------------------------
    # Find and identify all tiles
    # ------------------------------------------------------------

    tile_pattern = re.compile(
        r"tile_x(\d+)_y(\d+)\.tif$",
        re.IGNORECASE
    )

    tile_files = {}

    for file in input_folder.glob("*.tif"):
        match = tile_pattern.match(file.name)

        if match is None:
            continue

        x = int(match.group(1))
        y = int(match.group(2))

        tile_files[(x, y)] = file

    # Check that all expected tiles exist
    missing_tiles = []

    for y in range(1, max_y + 1):
        for x in range(1, max_x + 1):
            if (x, y) not in tile_files:
                missing_tiles.append((x, y))

    if missing_tiles:
        raise FileNotFoundError(
            f"Missing tiles: {missing_tiles}"
        )

    print(f"Found {len(tile_files)} tiles.")

    # ------------------------------------------------------------
    # Load images
    # ------------------------------------------------------------

    images = {}

    for key, file in sorted(tile_files.items()):
        image = tifffile.imread(file)

        if image.ndim == 2:
            pass

        elif image.ndim == 3 and image.shape[-1] in (3, 4):
            pass

        else:
            raise ValueError(
                f"Unsupported image shape for {file.name}: "
                f"{image.shape}"
            )

        images[key] = image

    # ------------------------------------------------------------
    # Check that all dimensions match
    # ------------------------------------------------------------

    first_image = images[(1, 1)]

    height = first_image.shape[0]
    width = first_image.shape[1]

    dtype = first_image.dtype

    for key, image in images.items():

        if image.shape != first_image.shape:
            raise ValueError(
                f"Tile {key} has shape {image.shape}, "
                f"but expected {first_image.shape}."
            )

    print(f"Tile size: {width} x {height} pixels")
    print(f"Data type: {dtype}")

    # ------------------------------------------------------------
    # Grayscale, high-pass filtered copies for registration
    # ------------------------------------------------------------

    MIN_CORR = 0.2  # matches below this correlation are not trusted

    def make_gray(image):
        image = image.astype(np.float32)
        if image.ndim == 2:
            return image
        # RGB / RGBA: luminance from the first three channels
        return (
            0.2126 * image[..., 0] +
            0.7152 * image[..., 1] +
            0.0722 * image[..., 2]
        )

    def prepare(image):
        # Subtracting a blurred copy removes the vignetting, bright spot
        # and corner glare shared by every tile, which otherwise dominate
        # the correlation and produce confident-but-wrong matches.
        gray = make_gray(image)
        if highpass_sigma:
            gray -= gaussian_filter(gray, highpass_sigma)
        return gray

    tiles = [
        (x, y)
        for y in range(1, max_y + 1)
        for x in range(1, max_x + 1)
    ]
    tile_index = {tile: i for i, tile in enumerate(tiles)}
    n_tiles = len(tiles)

    pairs = {
        "horizontal": [((x, y), (x + 1, y)) for y in range(1, max_y + 1) for x in range(1, max_x)],
        "vertical":   [((x, y), (x, y + 1)) for y in range(1, max_y) for x in range(1, max_x + 1)],
    }

    # ------------------------------------------------------------
    # Expected overlap and search radii
    # ------------------------------------------------------------

    overlap_x = int(round(width * overlap_fraction))
    overlap_y = int(round(height * overlap_fraction))

    search_x = int(round(width * search_fraction))
    search_y = int(round(height * search_fraction))

    print(f"Expected horizontal overlap: {overlap_x} pixels")
    print(f"Expected vertical overlap:   {overlap_y} pixels")

    # Nominal neighbour steps implied by overlap_fraction
    nominal = {
        "horizontal": (width - overlap_x, 0.0),
        "vertical":   (0.0, height - overlap_y),
    }

    # ------------------------------------------------------------
    # Pairwise registration (symmetric window around the expected step)
    # ------------------------------------------------------------

    measurements = {"horizontal": [], "vertical": []}

    if refine or stage_step is None:

        if stage_step is None:
            # Search around the nominal step: narrow along the overlap
            # (the overlap size is roughly known), wide across it (the
            # camera may be slightly rotated relative to the stage).
            expected = nominal
            radii = {
                "horizontal": (min(overlap_x // 3, search_x), search_y),
                "vertical":   (search_x, min(overlap_y // 3, search_y)),
            }
        else:
            # Search tightly around the given stage step
            # (radius along the overlap is capped at a third of the overlap)
            expected = {"horizontal": stage_step[0], "vertical": stage_step[1]}
            r = max(2 * max_dev, 8)
            r_along_x = min(r, int(width - abs(stage_step[0][0])) // 3)
            r_along_y = min(r, int(height - abs(stage_step[1][1])) // 3)
            radii = {"horizontal": (r_along_x, r), "vertical": (r, r_along_y)}

        print("Preparing tiles for registration...")
        gray_images = {key: prepare(image) for key, image in images.items()}

        for direction in ("horizontal", "vertical"):
            for tile_a, tile_b in pairs[direction]:
                dx, dy, correlation, at_edge = register_pair(
                    gray_images[tile_a],
                    gray_images[tile_b],
                    *expected[direction],
                    *radii[direction],
                )
                trusted = (not at_edge) and correlation >= MIN_CORR
                measurements[direction].append(
                    (tile_a, tile_b, dx, dy, correlation, trusted)
                )
                if verbose:
                    print(
                        f"{direction:<10} {tile_a} -> {tile_b}: "
                        f"dx = {dx}, dy = {dy}, "
                        f"correlation = {correlation:.3f}"
                        f"{'' if trusted else '  (not trusted)'}"
                    )

        del gray_images  # free memory before blending

    # ------------------------------------------------------------
    # Stage step (the grid prior)
    # ------------------------------------------------------------

    if stage_step is None:
        steps = []
        for direction in ("horizontal", "vertical"):
            step, n_agree = consensus_step(measurements[direction], tol=max_dev)
            if step is None or n_agree < 2:
                print(
                    f"WARNING: could not estimate the {direction} step; "
                    f"using nominal {nominal[direction]}"
                )
                step = nominal[direction]
            print(
                f"Estimated {direction} step: dx = {step[0]:.1f}, dy = {step[1]:.1f} "
                f"({n_agree}/{len(measurements[direction])} pairs agree)"
            )
            steps.append(step)
        stage_step = tuple(steps)
        print(
            "  (pass stage_step=(({:.0f}, {:.0f}), ({:.0f}, {:.0f})) to reuse this)".format(
                *stage_step[0], *stage_step[1]
            )
        )

    step_of = {"horizontal": stage_step[0], "vertical": stage_step[1]}

    # ------------------------------------------------------------
    # Solve global tile positions
    # ------------------------------------------------------------

    #
    # Weighted least squares over:
    #   position_b - position_a = stage step   (every pair, weight prior_weight)
    #   position_b - position_a = measurement  (good matches only, weight 1)
    # with tile (1, 1) fixed at (0, 0).
    #

    rows, values_x, values_y, weights = [], [], [], []

    def add_equation(tile_a, tile_b, dx, dy, weight):
        row = np.zeros(n_tiles)
        row[tile_index[tile_b]] = 1
        row[tile_index[tile_a]] = -1
        rows.append(row)
        values_x.append(dx)
        values_y.append(dy)
        weights.append(weight)

    n_used = n_total = 0

    for direction in ("horizontal", "vertical"):
        step = step_of[direction]

        for tile_a, tile_b in pairs[direction]:
            add_equation(tile_a, tile_b, *step, prior_weight if refine else 1.0)

        if refine:
            for tile_a, tile_b, dx, dy, correlation, trusted in measurements[direction]:
                n_total += 1
                close_to_grid = max(abs(dx - step[0]), abs(dy - step[1])) <= max_dev
                if trusted and close_to_grid:
                    add_equation(tile_a, tile_b, dx, dy, 1.0)
                    n_used += 1

    if refine:
        print(
            f"Refinement: used {n_used}/{n_total} pairwise matches; "
            f"the rest follow the stage grid"
        )
    else:
        print("Placing tiles on the stage grid (no refinement)")

    # Anchor tile (1, 1)
    row = np.zeros(n_tiles)
    row[tile_index[(1, 1)]] = 1
    rows.append(row)
    values_x.append(0.0)
    values_y.append(0.0)
    weights.append(10.0)

    sqrt_w = np.sqrt(np.asarray(weights))
    A = np.vstack(rows) * sqrt_w[:, None]
    x_positions = np.linalg.lstsq(A, np.asarray(values_x) * sqrt_w, rcond=None)[0]
    y_positions = np.linalg.lstsq(A, np.asarray(values_y) * sqrt_w, rcond=None)[0]

    positions = {
        tile: (float(x_positions[i]), float(y_positions[i]))
        for i, tile in enumerate(tiles)
    }

    # ------------------------------------------------------------
    # Shift coordinates so the smallest values are zero
    # ------------------------------------------------------------

    min_x = min(pos[0] for pos in positions.values())
    min_y = min(pos[1] for pos in positions.values())

    positions = {
        tile: (
            pos[0] - min_x,
            pos[1] - min_y
        )
        for tile, pos in positions.items()
    }

    # ------------------------------------------------------------
    # Calculate output dimensions
    # ------------------------------------------------------------

    max_right = max(
        positions[tile][0] + width
        for tile in tiles
    )

    max_bottom = max(
        positions[tile][1] + height
        for tile in tiles
    )

    output_width = int(round(max_right))
    output_height = int(round(max_bottom))

    print(
        f"Final image size: "
        f"{output_width} x {output_height} pixels"
    )

    # ------------------------------------------------------------
    # Create feathering weights
    # ------------------------------------------------------------

    feather_x = min(overlap_x, width // 2)
    feather_y = min(overlap_y, height // 2)

    # Horizontal feathering ramps
    ramp_up_x = np.linspace(
        0,
        1,
        feather_x,
        endpoint=False,
        dtype=np.float32
    )

    ramp_down_x = ramp_up_x[::-1]

    # Vertical feathering ramps
    ramp_up_y = np.linspace(
        0,
        1,
        feather_y,
        endpoint=False,
        dtype=np.float32
    )

    ramp_down_y = ramp_up_y[::-1]

    # ------------------------------------------------------------
    # Create accumulation arrays
    # ------------------------------------------------------------

    is_colour = first_image.ndim == 3

    if is_colour:
        channels = first_image.shape[-1]

        accumulation = np.zeros(
            (output_height, output_width, channels),
            dtype=np.float32
        )

    else:
        accumulation = np.zeros(
            (output_height, output_width),
            dtype=np.float32
        )

    weight_accumulation = np.zeros(
        (output_height, output_width),
        dtype=np.float32
    )

    # ------------------------------------------------------------
    # Add each tile
    # ------------------------------------------------------------

    for tile in tiles:

        x, y = tile

        image = images[tile].astype(np.float32)

        pos_x, pos_y = positions[tile]

        x0 = int(round(pos_x))
        y0 = int(round(pos_y))

        x1 = x0 + width
        y1 = y0 + height

        # Start with no feathering
        wx = np.ones(width, dtype=np.float32)
        wy = np.ones(height, dtype=np.float32)

        # Feather left edge when another tile is to the left
        if x > 1:
            wx[:feather_x] = ramp_up_x

        # Feather right edge when another tile is to the right
        if x < max_x:
            wx[-feather_x:] = ramp_down_x

        # Feather top edge when another tile is above
        if y > 1:
            wy[:feather_y] = ramp_up_y

        # Feather bottom edge when another tile is below
        if y < max_y:
            wy[-feather_y:] = ramp_down_y

        weight = wy[:, None] * wx[None, :]

        if is_colour:
            accumulation[
                y0:y1,
                x0:x1,
                :
            ] += image * weight[..., None]

        else:
            accumulation[
                y0:y1,
                x0:x1
            ] += image * weight

        weight_accumulation[
            y0:y1,
            x0:x1
        ] += weight

    # ------------------------------------------------------------
    # Normalise
    # ------------------------------------------------------------

    valid = weight_accumulation > 0

    if is_colour:

        stitched = np.zeros_like(
            accumulation,
            dtype=np.float32
        )

        stitched[valid] = (
            accumulation[valid]
            /
            weight_accumulation[valid, None]
        )

    else:

        stitched = np.zeros_like(
            accumulation,
            dtype=np.float32
        )

        stitched[valid] = (
            accumulation[valid]
            /
            weight_accumulation[valid]
        )

    # ------------------------------------------------------------
    # Convert back to original data type
    # ------------------------------------------------------------

    if np.issubdtype(dtype, np.integer):

        info = np.iinfo(dtype)

        stitched = np.clip(
            stitched,
            info.min,
            info.max
        ).astype(dtype)

    else:

        stitched = stitched.astype(dtype)

    # ------------------------------------------------------------
    # Save stitched image
    # ------------------------------------------------------------

    stem = Path(output_file).with_suffix("")

    if save_full_tiff:
        save_tiff(stem.with_suffix(".tif"), stitched, pixel_size_um)

    low = downsample(stitched, factor, method)
    low_px = None if pixel_size_um is None else pixel_size_um * factor

    if factor > 1 and save_lowres_tiff:
        save_tiff(stem.parent / f"{stem.name}_x{factor}.tif", low, low_px)

    if save_png:
        from PIL import Image
        png_path = stem.parent / f"{stem.name}_x{factor}.png"
        Image.fromarray(make_display_image(low)).save(png_path)
        print(f"Saved PNG to: {png_path}")

    if low_px is not None:
        print(f"Plate scale: {pixel_size_um} um/px -> {low_px} um/px (factor {factor})")

    return stitched, positions, low