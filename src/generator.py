from typing import List
import numpy as np
from src.dsl import Hypothesis
from src.perception import grid_attributes

def generate_hypotheses(examples: List[dict]) -> List[Hypothesis]:
    """
    Seed initial hypothesis pool from:
    1. Structural diff between input/output grids
    2. Object-level attribute changes
    3. Symmetry/count observations
    """
    hyps = []
    hid = 0

    for ex in examples:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        in_attrs  = grid_attributes(inp)
        out_attrs = grid_attributes(out)

        # Shape change? → scaling or cropping hypothesis
        if inp.shape != out.shape:
            hyps.append(Hypothesis(
                id=f"h{hid:03d}", description="output is scaled/cropped version of input",
                condition="always", transform=f"resize_to({out.shape})",
                support=1))
            hid += 1

        # Color mapping? → recolor hypotheses
        in_colors  = set(int(v) for v in inp.flatten())
        out_colors = set(int(v) for v in out.flatten())
        if in_colors != out_colors:
            added = out_colors - in_colors
            removed = in_colors - out_colors
            if len(added)==1 and len(removed)==1:
                hyps.append(Hypothesis(
                    id=f"h{hid:03d}",
                    description=f"recolor {list(removed)[0]} → {list(added)[0]}",
                    condition=f"color_is({list(removed)[0]})",
                    transform=f"recolor({list(removed)[0]},{list(added)[0]})",
                    support=1))
                hid += 1

        # Object count change? → copy/delete hypothesis
        dn = out_attrs["num_objects"] - in_attrs["num_objects"]
        if dn > 0:
            hyps.append(Hypothesis(
                id=f"h{hid:03d}",
                description=f"output has {dn} more objects — copy/replicate rule",
                condition="is_largest", transform=f"copy_{dn}_times",
                support=1))
            hid += 1
        elif dn < 0:
            hyps.append(Hypothesis(
                id=f"h{hid:03d}",
                description=f"output has {abs(dn)} fewer objects — deletion rule",
                condition="is_smallest", transform="delete",
                support=1))
            hid += 1

        # Symmetry appeared? → mirror hypothesis
        if out_attrs["has_symmetry_h"] and not in_attrs["has_symmetry_h"]:
            hyps.append(Hypothesis(
                id=f"h{hid:03d}", description="flip_h to create symmetry",
                condition="always", transform="flip_h_and_merge", support=1))
            hid += 1

    # Deduplicate by description
    seen = set()
    unique = []
    for h in hyps:
        if h.description not in seen:
            seen.add(h.description)
            unique.append(h)
    return unique
