from dataclasses import dataclass

DSL_CONDITIONS = {
    "is_largest":   lambda objs, o: o == max(objs, key=lambda x: x.size),
    "is_smallest":  lambda objs, o: o == min(objs, key=lambda x: x.size),
    "is_rect":      lambda objs, o: o.is_rectangle,
    "is_square":    lambda objs, o: o.is_rectangle and o.height == o.width,
    "color_is":     lambda objs, o, c: o.color == c,
    "size_gt":      lambda objs, o, n: o.size > n,
    "count_color":  lambda objs, o, c: sum(1 for x in objs if x.color==c),
}

DSL_TRANSFORMS = {
    "recolor":     "change object color to c",
    "move_to":     "translate object by (dr, dc)",
    "rotate_90_cw": "rotate object 90 degrees CW",
    "rotate_180":  "rotate object 180 degrees",
    "flip_horizontal": "flip object horizontally",
    "flip_vertical": "flip object vertically",
    "scale_up":    "scale object by factor f",
    "delete_smallest_object": "remove smallest object from grid",
    "delete_largest_object": "remove largest object from grid",
    "gravity_down": "objects fall to bottom",
    "gravity_up": "objects fall to top",
    "tile_2x2": "repeat pattern 2x2",
    "crop_to_bounding_box": "crop to bounding box",
    "fill_holes": "flood-fill interior",
    "outline_only": "keep perimeter cells",
    "sort_rows_by_color": "sort rows by color",
    "mirror_and_append": "create symmetry"
}

@dataclass
class Hypothesis:
    id: str
    description: str
    condition: str       # DSL condition name + args
    transform: str       # DSL transform name + args
    confidence: float = 1.0
    support: int = 0     # number of examples consistent with this
    falsified: bool = False

    def __repr__(self):
        return f"H[{self.id}](conf={self.confidence:.2f}): IF {self.condition} THEN {self.transform}"
