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

# ── Clef types ────────────────────────────────────────────────────────────────
CLEF_TREBLE = "treble"
CLEF_BASS   = "bass"
CLEF_ALTO   = "alto"
CLEF_TENOR  = "tenor"
CLEF_UNKNOWN = "unknown"

# Fraction of staff width reserved for the clef region (left side)
CLEF_REGION_RATIO = 0.12

# ── Pitch constants ───────────────────────────────────────────────────────────
# Diatonic pitch names (C=0 … B=6)
PITCH_NAMES = ["C", "D", "E", "F", "G", "A", "B"]

# Base pitch at the bottom staff line for each clef type.
# Encoded as (pitch_name_index, octave) so we don't need a music library.
#   Treble: bottom line = E4  →  PITCH_NAMES[2]=E, octave 4
#   Bass:   bottom line = G2  →  PITCH_NAMES[4]=G, octave 2
#   Alto:   bottom line = F3  →  PITCH_NAMES[3]=F, octave 3
#   Tenor:  bottom line = D3  →  PITCH_NAMES[1]=D, octave 3
CLEF_BOTTOM_LINE_PITCH: dict = {
    CLEF_TREBLE:  (2, 4),   # E4
    CLEF_BASS:    (4, 2),   # G2
    CLEF_ALTO:    (3, 3),   # F3
    CLEF_TENOR:   (1, 3),   # D3
    CLEF_UNKNOWN: (2, 4),   # default to treble
}

# ── Duration constants ────────────────────────────────────────────────────────
# Duration string names (subset of MusicXML durationType)
DUR_WHOLE      = "whole"        # open head, no stem
DUR_HALF       = "half"         # open head, stem
DUR_QUARTER    = "quarter"      # filled head, stem, 0 flags
DUR_EIGHTH     = "eighth"       # filled head, stem, 1 flag
DUR_SIXTEENTH  = "16th"         # filled head, stem, 2 flags
DUR_32ND       = "32nd"         # filled head, stem, 3 flags

# Duration in quarter-note beats
DUR_BEATS: dict = {
    DUR_WHOLE:     4.0,
    DUR_HALF:      2.0,
    DUR_QUARTER:   1.0,
    DUR_EIGHTH:    0.5,
    DUR_SIXTEENTH: 0.25,
    DUR_32ND:      0.125,
}

# Thresholds for open/closed head classification
# Ratio of filled pixels inside the notehead ellipse
OPEN_HEAD_FILL_RATIO  = 0.40   # below this → open (half/whole)
CLOSED_HEAD_FILL_RATIO = 0.65  # above this → closed (quarter/eighth…)

# Stem detection: minimum stem length as multiple of staff unit
STEM_MIN_LENGTH_RATIO = 1.0    # stem must be at least 1× staff unit long
# Search margin on each side of notehead for stem (in fractions of staff unit)
STEM_SEARCH_MARGIN_RATIO = 0.55

# Flag/beam detection: minimum flag height (in fractions of staff unit)
FLAG_MIN_HEIGHT_RATIO = 0.45

# ── Barline constants ─────────────────────────────────────────────────────────
BARLINE_MIN_HEIGHT_RATIO = 2.5   # must span ≥ 2.5× staff unit height
BARLINE_MAX_WIDTH_PX     = 8     # thin barline is ≤ 8 px wide
BARLINE_MERGE_TOL_PX     = 10    # merge barline fragments within 10 px
