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
        if "recolor" in hyp.transform:
            # parse: recolor(from_c, to_c)
            parts = hyp.transform.replace("recolor(","").replace(")","").split(",")
            if len(parts) == 2:
                fc, tc = int(parts[0]), int(parts[1])
                result[result == fc] = tc
            elif len(parts) == 1:
                tc = int(parts[0])
                # recolor all non-zero
                result[result != 0] = tc

        elif "flip_h" in hyp.transform:
            result = np.fliplr(result)

        elif "flip_v" in hyp.transform:
            result = np.flipud(result)

        elif "rotate_90" in hyp.transform:
            result = np.rot90(result, k=1)

        elif "delete" in hyp.transform:
            # delete smallest object
            if objects:
                smallest = min(objects, key=lambda o: o.size)
                for r,c in smallest.cells:
                    result[r,c] = 0

        elif "gravity" in hyp.transform:
            # simplified: fall down
            for col in range(result.shape[1]):
                column = result[:,col]
                non_zero = column[column != 0]
                zeros = column[column == 0]
                result[:,col] = np.concatenate([zeros, non_zero])

        elif "resize_to" in hyp.transform:
            # extract target shape from string
            import re
            match = re.search(r'\((\d+),\s*(\d+)\)', hyp.transform)
            if match:
                h, w = int(match.group(1)), int(match.group(2))
                from scipy.ndimage import zoom
                factors = (h/result.shape[0], w/result.shape[1])
                result = zoom(result, factors, order=0).astype(int)

        return result
    except Exception:
        return None
