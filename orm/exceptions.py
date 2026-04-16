"""OMR-specific exception hierarchy."""


# ── Base ──────────────────────────────────────────────────────────────────────
class OrmException(Exception):
    """Root exception for all OMR errors."""


# ── Image loading ─────────────────────────────────────────────────────────────
class ImageLoadError(OrmException):
    """Raised when an input image cannot be read."""


# ── Model / inference ─────────────────────────────────────────────────────────
class ModelNotFoundError(OrmException):
    """Raised when a required checkpoint file is missing."""


# ── Staffline extraction ──────────────────────────────────────────────────────
class StafflineException(OrmException):
    """Base class for staffline-related errors."""


class StafflineNotDetected(StafflineException):
    """Raised when no staff lines are found in the image."""


class StafflineCountInconsistent(StafflineException):
    """Raised when the detected number of stafflines is unexpected."""


# ── Notehead extraction ───────────────────────────────────────────────────────
class NoteheadException(OrmException):
    """Base class for notehead-related errors."""
