import numpy as np
from typing import Optional
from src.dsl import Hypothesis
from src.perception import extract_objects

def apply_hypothesis(hyp: Hypothesis, grid: np.ndarray) -> Optional[np.ndarray]:
    """
    Execute DSL program represented by hypothesis on a grid.
    Returns predicted output, or None if hypothesis is inapplicable.
    """
    result = grid.copy()
    objects = extract_objects(result)

    try:
        t = hyp.transform
        if "recolor" in t:
            # parse: recolor(from_c, to_c)
            parts = t.replace("recolor(","").replace(")","").split(",")
            if len(parts) == 2:
                fc, tc = int(parts[0]), int(parts[1])
                result[result == fc] = tc
            elif len(parts) == 1:
                tc = int(parts[0])
                result[result != 0] = tc

        elif "flip_horizontal" in t or "flip_h" in t:
            result = np.fliplr(result)

        elif "flip_vertical" in t or "flip_v" in t:
            result = np.flipud(result)

        elif "rotate_90_cw" in t or "rotate_90" in t:
            result = np.rot90(result, k=-1) # CW rotation

        elif "rotate_180" in t:
            result = np.rot90(result, k=2)

        elif "delete_smallest_object" in t:
            if objects:
                smallest = min(objects, key=lambda o: o.size)
                for r,c in smallest.cells:
                    result[r,c] = 0

        elif "delete_largest_object" in t:
            if objects:
                largest = max(objects, key=lambda o: o.size)
                for r,c in largest.cells:
                    result[r,c] = 0

        elif "gravity_down" in t:
            for col in range(result.shape[1]):
                column = result[:,col]
                non_zero = column[column != 0]
                zeros = column[column == 0]
                result[:,col] = np.concatenate([zeros, non_zero])

        elif "gravity_up" in t:
            for col in range(result.shape[1]):
                column = result[:,col]
                non_zero = column[column != 0]
                zeros = column[column == 0]
                result[:,col] = np.concatenate([non_zero, zeros])

        elif "tile_2x2" in t:
            result = np.tile(result, (2, 2))

        elif "crop_to_bounding_box" in t:
            if objects:
                rmin = min(o.bbox[0] for o in objects)
                cmin = min(o.bbox[1] for o in objects)
                rmax = max(o.bbox[2] for o in objects)
                cmax = max(o.bbox[3] for o in objects)
                result = result[rmin:rmax+1, cmin:cmax+1]

        elif "fill_holes" in t:
            from scipy.ndimage import binary_fill_holes
            for color in np.unique(result):
                if color == 0: continue
                mask = (result == color)
                filled = binary_fill_holes(mask)
                result[filled] = color

        elif "outline_only" in t:
            from scipy.ndimage import binary_erosion
            for color in np.unique(result):
                if color == 0: continue
                mask = (result == color)
                eroded = binary_erosion(mask)
                result[mask & ~eroded] = color
                result[eroded] = 0

        elif "sort_rows_by_color" in t:
            # Sort each row independently based on color values
            result = np.sort(result, axis=1)

        elif "mirror_and_append" in t:
            # Append horizontally mirrored copy to the right
            flipped = np.fliplr(result)
            result = np.concatenate([result, flipped], axis=1)

        elif "resize_to" in t:
            import re
            match = re.search(r'\((\d+),\s*(\d+)\)', t)
            if match:
                h, w = int(match.group(1)), int(match.group(2))
                from scipy.ndimage import zoom
                factors = (h/result.shape[0], w/result.shape[1])
                result = zoom(result, factors, order=0).astype(int)

        return result
    except Exception:
        return None
