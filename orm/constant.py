"""Pipeline-wide constants for the OMR system."""

# ── Staff detection ───────────────────────────────────────────────────────────
STAFF_LINE_COUNT = 5        # number of lines per staff system

# ── Model patch sizes ─────────────────────────────────────────────────────────
STAFFLINE_PATCH_SIZE = 256  # input patch size for stream-1 (unet_big)
SEMANTIC_PATCH_SIZE = 288   # input patch size for stream-2 (seg_net)

# ── Channel indices — stream 1 (staffline model) ─────────────────────────────
M1_CH_BG = 0
M1_CH_STAFF = 1
M1_CH_SYMBOL = 2

# ── Channel indices — stream 2 (semantic model) ──────────────────────────────
M2_CH_BG = 0
M2_CH_NOTEHEAD = 1
M2_CH_STEM = 2
M2_CH_SYMBOL = 3

# ── Inference thresholds ─────────────────────────────────────────────────────
STAFF_CONF_THRESH = 0.3
NOTE_CONF_THRESH = 0.4
SYMBOL_CONF_THRESH = 0.35

# ── Tile-and-stitch ───────────────────────────────────────────────────────────
DEFAULT_OVERLAP = 64
DEFAULT_MAX_SIDE = 2048

# ── Notehead filter ───────────────────────────────────────────────────────────
NOTEHEAD_MIN_AREA = 12
NOTEHEAD_MAX_AREA = 2000
NOTEHEAD_ASPECT_RATIO = (0.35, 2.0)
