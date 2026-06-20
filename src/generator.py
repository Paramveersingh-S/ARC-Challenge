from typing import List
import numpy as np
from src.dsl import Hypothesis
from src.perception import grid_attributes

def generate_hypotheses(examples: List[dict]) -> List[Hypothesis]:
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

        # Global Color Mapping (deduce full palette mapping from input->output)
        if inp.shape == out.shape:
            mapping = {}
            valid = True
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    ic, oc = int(inp[i,j]), int(out[i,j])
                    if ic != oc:
                        if ic not in mapping:
                            mapping[ic] = oc
                        elif mapping[ic] != oc:
                            valid = False
            if valid and mapping:
                map_str = ",".join(f"{k}:{v}" for k,v in mapping.items())
                hyps.append(Hypothesis(
                    id=f"h{hid:03d}", description=f"map colors {map_str}",
                    condition="always", transform=f"map_colors({map_str})", support=0))
                hid += 1
                
    seen = set()
    unique = []
    for h in hyps:
        if h.transform not in seen:
            seen.add(h.transform)
            unique.append(h)
            
    return unique
