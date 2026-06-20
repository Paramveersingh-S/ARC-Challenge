from typing import List
import numpy as np
from src.dsl import Hypothesis
from src.perception import grid_attributes

def generate_hypotheses(examples: List[dict]) -> List[Hypothesis]:
    """
    Seed initial hypothesis pool from:
    1. Generic, parameter-less DSL transformations (always added).
    2. Structural diff between input/output grids.
    3. Object-level attribute changes.
    """
    hyps = []
    hid = 0

    # 1. Base Generic Transformations
    generic_transforms = [
        "flip_horizontal", "flip_vertical", "rotate_90_cw", "rotate_180",
        "gravity_down", "gravity_up", "tile_2x2", "fill_holes", "outline_only",
        "sort_rows_by_color", "mirror_and_append"
    ]
    for t in generic_transforms:
        hyps.append(Hypothesis(
            id=f"h{hid:03d}", description=f"Apply {t} globally",
            condition="always", transform=t, support=0
        ))
        hid += 1

    # 2. Extract Specific Parameterized Heuristics
    for ex in examples:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        in_attrs  = grid_attributes(inp)
        out_attrs = grid_attributes(out)

        if inp.shape != out.shape:
            hyps.append(Hypothesis(
                id=f"h{hid:03d}", description="output is scaled/cropped version of input",
                condition="always", transform=f"resize_to({out.shape})", support=0))
            hid += 1

        in_colors  = set(int(v) for v in inp.flatten() if v != in_attrs['bg_color'])
        out_colors = set(int(v) for v in out.flatten() if v != out_attrs['bg_color'])
        
        if in_colors != out_colors:
            added = out_colors - in_colors
            removed = in_colors - out_colors
            if len(added)==1 and len(removed)==1:
                hyps.append(Hypothesis(
                    id=f"h{hid:03d}",
                    description=f"recolor {list(removed)[0]} → {list(added)[0]}",
                    condition=f"color_is({list(removed)[0]})",
                    transform=f"recolor({list(removed)[0]},{list(added)[0]})",
                    support=0))
                hid += 1
            elif len(added)==1 and len(removed)==0:
                # Color everything to the added color
                hyps.append(Hypothesis(
                    id=f"h{hid:03d}",
                    description=f"recolor all to {list(added)[0]}",
                    condition="always",
                    transform=f"recolor({list(added)[0]})",
                    support=0))
                hid += 1

    # Deduplicate by transform signature
    seen = set()
    unique = []
    for h in hyps:
        if h.transform not in seen:
            seen.add(h.transform)
            unique.append(h)
            
    return unique
