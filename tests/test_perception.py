import numpy as np
from src.perception import extract_objects, grid_attributes

def test_extract_objects():
    grid = np.array([
        [0, 1, 1],
        [0, 1, 0],
        [2, 0, 0]
    ])
    objs = extract_objects(grid, background=0)
    assert len(objs) == 2
    colors = {o.color for o in objs}
    assert 1 in colors
    assert 2 in colors

def test_grid_attributes():
    grid = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])
    attrs = grid_attributes(grid)
    assert attrs["num_colors"] == 1
    assert attrs["has_symmetry_h"] == True
    assert attrs["has_symmetry_v"] == True
