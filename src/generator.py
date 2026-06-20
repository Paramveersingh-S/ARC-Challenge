from typing import List
import numpy as np
from src.dsl import Hypothesis
from src.perception import grid_attributes
from src.llm_coder import generate_solve_function

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

        shape_changed = (inp.shape != out.shape)
        resize_transform = None

        if shape_changed:
            resize_transform = f"resize_to({out.shape})"
            hyps.append(Hypothesis(
                id=f"h{hid:03d}", description="output is scaled/cropped version of input",
                condition="always", transform=resize_transform, support=0))
            hid += 1

        # Global Color Mapping (deduce full palette mapping from input->output)
        mapping_str = None
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
                mapping_str = f"map_colors({map_str})"
                hyps.append(Hypothesis(
                    id=f"h{hid:03d}", description=f"map colors {map_str}",
                    condition="always", transform=mapping_str, support=0))
                hid += 1
                
                # Combinatorial: Map colors AND generic transform
                for t in ["rotate_90_cw", "flip_horizontal", "flip_vertical"]:
                    hyps.append(Hypothesis(
                        id=f"h{hid:03d}", description=f"{t} then map colors",
                        condition="always", transform=f"{t} | {mapping_str}", support=0))
                    hid += 1

        # Fast Object-Specific Rules
        if not shape_changed:
            in_colors = set(int(v) for v in inp.flatten())
            out_colors = set(int(v) for v in out.flatten())
            if len(in_colors) > len(out_colors):
                missing = list(in_colors - out_colors)
                if len(missing) == 1:
                    hyps.append(Hypothesis(
                        id=f"h{hid:03d}", description=f"delete objects of color {missing[0]}",
                        condition="always", transform=f"delete_color({missing[0]})", support=0))
                    hid += 1
                    
        # Object bounds logic
        if in_attrs['num_objects'] > 0:
            hyps.append(Hypothesis(
                id=f"h{hid:03d}", description="recolor largest object to output majority",
                condition="always", transform="recolor_largest_to_majority", support=0))
            hid += 1

    # 3. LLM Code Synthesis Engine (Phase 7)
    llm_code = generate_solve_function(examples)
    if llm_code and "def solve(" in llm_code:
        hyps.append(Hypothesis(
            id=f"h{hid:03d}", description="LLM Synthesized Python Logic",
            condition="always", transform=llm_code, support=0))
        hid += 1

    seen = set()
    unique = []
    for h in hyps:
        if h.transform not in seen:
            seen.add(h.transform)
            unique.append(h)
            
    return unique
