# Notebook revision index

All changes on this branch require author review before merging into main.

| Analysis | File | Status |
| --- | --- | --- |
| Catalog detectability | `LDA-plots.ipynb` | Updated from desktop source; tested with the September 28 CSV. See LDA_REVIEW.md. |
| Longer-window visual review | `visibilityofseismicity_reviewer_100km.ipynb` | Added local documented source; configured for Aug 25-Sep 18, not the complete study interval. Requires server data. |
| Amplitudes and geometry | `plotroad_plusamplitudes_revision_review.ipynb` | Added annotated working copy; legacy labels and noise-distance correction still require reconciliation with final figures. Does not replace the older notebook yet. |
| Station template matching | `../template-matching/templates-seismometers_review.ipynb` | Annotated exploratory source, not a verified five-station production pipeline. |
| DAS pick-quality criteria | `../ML/DAS_pick_quality_workflow.ipynb` | New step-by-step entry point for the existing catalog script; requires pick CSVs. |

The ML notebook intentionally uses placeholder data paths and refuses to reuse
an existing output directory. It is not expected to run before the user supplies
input/output locations. Alex's station ELEP/PyOcto workflow is still to be
contributed separately, preferably through its own pull request.

No full waveform reprocessing or new manual event classifications were performed
for these additions. Code cells in the two imported working notebooks are
preserved; documentation describes unresolved discrepancies rather than silently
changing scientific settings. The older amplitude notebook remains available.
