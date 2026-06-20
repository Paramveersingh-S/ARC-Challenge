import numpy as np
from typing import List, Tuple, Set
from dataclasses import dataclass

@dataclass
class ARCObject:
    """A connected region of same-color cells."""
    color: int
    cells: List[Tuple[int,int]]
    bbox: Tuple[int,int,int,int]   # row_min, col_min, row_max, col_max
    shape_signature: frozenset      # canonical shape hash

    @property
    def size(self): return len(self.cells)

    @property
    def height(self): return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self): return self.bbox[3] - self.bbox[1] + 1

    @property
    def is_rectangle(self):
        return self.size == self.height * self.width

    @property
    def center(self):
        rows = [c[0] for c in self.cells]
        cols = [c[1] for c in self.cells]
        return (sum(rows)/len(rows), sum(cols)/len(cols))

def infer_background(grid: np.ndarray) -> int:
    """Dynamically infer background color. Mostly 0, but fallback to edge frequency."""
    if 0 in grid:
        return 0
    # Check edges
    edges = np.concatenate([grid[0,:], grid[-1,:], grid[:,0], grid[:,-1]])
    unique, counts = np.unique(edges, return_counts=True)
    if len(counts) > 0:
        return int(unique[np.argmax(counts)])
    return 0

def extract_objects(grid: np.ndarray, background: int = -1, group_by_color: bool = False) -> List[ARCObject]:
    """BFS connected-component extraction, strict 4-connectivity.
       If group_by_color is True, disjoint regions of the same color are treated as one object.
    """
    if background == -1:
        background = infer_background(grid)
        
    visited = np.zeros_like(grid, dtype=bool)
    objects = []
    
    if group_by_color:
        colors = np.unique(grid)
        for c in colors:
            if c == background: continue
            cells = []
            for r in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    if grid[r,col] == c:
                        cells.append((r, col))
                        visited[r,col] = True
            if cells:
                rows = [x[0] for x in cells]
                cols = [x[1] for x in cells]
                bbox = (min(rows), min(cols), max(rows), max(cols))
                r0,c0 = min(rows), min(cols)
                sig = frozenset((r-r0, c-c0) for r,c in cells)
                objects.append(ARCObject(int(c), cells, bbox, sig))
        return objects

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if visited[r,c] or grid[r,c] == background:
                continue
            color = grid[r,c]
            cells = []
            queue = [(r,c)]
            while queue:
                cr, cc = queue.pop()
                if visited[cr,cc]: continue
                visited[cr,cc] = True
                cells.append((cr,cc))
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc = cr+dr, cc+dc
                    if 0<=nr<grid.shape[0] and 0<=nc<grid.shape[1]:
                        if not visited[nr,nc] and grid[nr,nc]==color:
                            queue.append((nr,nc))
            rows = [x[0] for x in cells]
            cols = [x[1] for x in cells]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            # Canonical shape: normalize to origin
            r0,c0 = min(rows), min(cols)
            sig = frozenset((r-r0, c-c0) for r,c in cells)
            objects.append(ARCObject(color, cells, bbox, sig))
    return objects

def grid_attributes(grid: np.ndarray) -> dict:
    """High-level grid-level attributes for hypothesis seeding."""
    bg = infer_background(grid)
    objs = extract_objects(grid, background=bg)
    colors = set(int(v) for v in grid.flatten() if v != bg)
    return {
        "shape": grid.shape,
        "bg_color": bg,
        "num_colors": len(colors),
        "colors": sorted(colors),
        "num_objects": len(objs),
        "objects": objs,
        "has_symmetry_h": np.array_equal(grid, np.fliplr(grid)),
        "has_symmetry_v": np.array_equal(grid, np.flipud(grid)),
        "has_diagonal_sym": (grid.shape[0]==grid.shape[1] and
                              np.array_equal(grid, grid.T)),
    }
