# Template-matching versions under review

This branch stages local files for author review. It does not certify that every
file is the production version used for the revised manuscript.

## Files

- `main_template_matching_algorithm.py`: unchanged historical driver (short August
  window, 3-second templates, 2-9.8 Hz). Retained until its replacement is approved.
- `main_template_matching_algorithm-julian_review.py`: local Downloads candidate
  with the August 23, 2023 21:00 to November 23, 2023 21:00 UTC matching window,
  5-second templates, channels `[0, 2500)`, 50 Hz input and a 2-10 Hz filter.
  Its template query covers August 25-31 and a 35-km radius. These settings are
  preserved, not changed to match manuscript wording. Server paths and matching
  helper-module versions still require verification. The multiprocessing code
  is not a portable, tested entry point and must not be imported to inspect it.
- `templates-seismometers_review.ipynb`: local Rainier_notebook working notebook,
  with original code preserved and outputs cleared. It mixes 11-station template
  preparation, four-West examples, and loading saved three-month result CSVs.
  This does not establish which stations produced the final catalog. Its last
  plot hardcodes 30 unmatched station detections, before regional screening.

## Before merging as final methods

1. Confirm the actual DAS production driver and helper versions from the server.
2. Reconcile the 2-10 Hz filter and 35-km template query with the manuscript.
3. Obtain the five-station TM production configuration; do not infer it from
   example station lists or rewrite historical lists without evidence.
4. Trace the final 85 DAS and 92 station detections to their saved source CSVs and
   the subsequent regional-screening/manual-review tables.
5. Run a small waveform example with the required data before a full rerun.

No raw-data processing or network waveform downloads were run for this update.
