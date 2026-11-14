"""trained-reader-audit: a compute-matched, controlled study of a trained reader.

The grader, extractor, seeds, store and token accounting come from the closed audit
(``channel_audit``, tag ``audit-final``); nothing of them is copied here.
"""

from __future__ import annotations

__version__ = "0.0.1"

# The frozen round-0 prompt of the audit. The dev pools were generated with it, and so
# will every candidate pool of this study, so its hash is pinned here and in a test.
ROUND0_PROMPT_SHA256 = (
    "42ab28d01ae379edf347d9818800bcb1c7cd8ec6985c484a20e0850df2e96189"
)


def _require_audit() -> None:
    try:
        from channel_audit import channels
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ImportError(
            "reader_audit needs mas-channel-audit at the tag audit-final; "
            "see the Setup section of README.md"
        ) from exc
    if not channels.PROMPTS_DIR.is_dir():  # pragma: no cover - environment problem
        raise ImportError(
            "channel_audit was installed as a wheel, so its frozen prompts are "
            "missing; install it as an editable checkout (see README.md)"
        )


_require_audit()
