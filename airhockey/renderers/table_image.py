"""Which table bitmap the renderers draw under the puck and paddle.

Every renderer (``AirHockeyRenderer``, the real-robot sim view, the camera overlay)
stretches this image corner to corner over the Box2D table rectangle, so the
bitmap has to be a top-down picture of the playing surface in the table frame:

    stored PNG row r    <->  table x = -L/2 + (r + 0.5) / ppm   (row 0 = far wall, last row = robot wall)
    stored PNG column c <->  table y = -W/2 + (c + 0.5) / ppm

(the renderers rotate it 90 degrees clockwise before use).

* ``air_hockey_table_real.png`` — the real table, made by
  ``scripts/real/capture_real_table_image.py`` from the calibrated camera, so its
  markings sit where the puck detector reports them. Used whenever it exists.
* ``air_hockey_table.png`` — the original stock rink drawing. Its markings do not
  match the real table (face-off circles ~10 cm too close to the centre line, end
  arcs that the real table does not have, borders drawn inside the walls).

Override with the environment variable ``AIRHOCKEY_TABLE_IMAGE``: ``real``,
``stock``, or a path to a PNG.
"""

import os
from pathlib import Path

STOCK_TABLE_IMAGE = "air_hockey_table.png"
REAL_TABLE_IMAGE = "air_hockey_table_real.png"
TABLE_IMAGE_ENV_VAR = "AIRHOCKEY_TABLE_IMAGE"


def default_assets_dir():
    return Path(__file__).resolve().parents[2] / "assets"


def table_image_path(assets_dir=None, choice=None):
    """Path of the table bitmap to draw.

    ``choice``: ``"real"``, ``"stock"``, a path, or None (environment variable,
    else the real image when it exists, else the stock one).
    """
    folder = Path(assets_dir) if assets_dir is not None else default_assets_dir()
    if choice is None:
        choice = os.environ.get(TABLE_IMAGE_ENV_VAR) or None
    if choice is None:
        real = folder / REAL_TABLE_IMAGE
        return real if real.exists() else folder / STOCK_TABLE_IMAGE
    if choice == "real":
        return folder / REAL_TABLE_IMAGE
    if choice == "stock":
        return folder / STOCK_TABLE_IMAGE
    return Path(choice)
