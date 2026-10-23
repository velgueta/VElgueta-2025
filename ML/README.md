# Machine-learning workflows

## DAS: waveform picking and candidate selection are separate steps

1. `Phasepicking_MORA.py` applies pretrained ELEP component models to DAS waveforms
   and exports channel-level picks. The current copy has server-specific paths
   and a two-hour December 10, 2023 example configuration, not the full study run.
   It uses a 0.5-49 Hz Butterworth filter before downsampling from 200 to 100 Hz.
   ELEP and its external helper modules must be installed separately.
2. `das_event_catalog.py` reads existing S-pick CSVs and evaluates spatial
   coherence and physically plausible relative moveout. It does not run ELEP,
   filter waveforms, query ComCat, or use PyOcto. It uses only the Python standard
   library and includes cable geometry, so no additional geometry file is required.

From the repository root:

```bash
python3 ML/das_event_catalog.py --input-dir /path/to/pick_csvs --output-dir /path/to/results
```

Alternatively, place the script beside the pick files and run
`python3 das_event_catalog.py`. It searches recursively for `MORADAS_*_AR.csv`;
use `--pattern` for a different filename convention. Do not mix unrelated CSVs.

## Selection criteria in the preserved script

- Initial blocks: at least 10 consecutive sampled channels (stride 4), with picks
  spanning at most 1 second.
- Overlapping or adjacent blocks: deduplicated with a 2-second temporal tolerance;
  prefer more channels, then a shorter pick-time span.
- Local coherence: nine-pick moving-median trend, scaled MAD at most 0.10 s,
  at least 85% of centered residuals within 0.20 s, and at least 10 consecutive
  sampled inlier channels.
- Event grouping: the next block starts no more than 3 s after the latest pick
  in the current group. This is a grouping gap, not an allowed adjacent-channel
  moveout, and it does not require neighboring blocks. Chained groups may span
  longer than 3 s.
- Physical moveout: trial sources within 25 km of Mount Rainier, initial grid
  spacing 2.5 km, depths 0-20 km in 2-km steps and S velocities 2.5, 3.0, 3.5,
  and 4.0 km/s, followed by local refinement. A common time offset is fitted.
- At least one locally coherent block must also pass the physical test: scaled
  residual MAD at most 0.10 s, 95th-percentile absolute residual at most 0.25 s,
  and at least 85% of residuals within 0.20 s. One noisy segment does not by itself
  reject a candidate that passes on another segment.

These are empirical screening settings, not a validated earthquake-location
procedure. A ComCat match or a previously known location is not required.
The common offset handles a uniform clock error within a candidate, not arbitrary
time distortion. The source grid is local; it is not a 1000-km source search.

The preserved script allows channels through 3000. The manuscript's statement
that accepted detections occurred within 0-2500 describes results, not this input
limit. Do not silently change the limit to match that statement.

## Outputs and interpretation

`DAS_event_catalog.csv` contains candidate classifications and metrics. Supporting
block, pick and file-summary CSVs and `DAS_run_configuration.json` retain the
processing details. Classes are `coherent`, `coherent_low_support`, and `rejected`;
the strong-support threshold is 20 selected picks. Passing is not proof of an
earthquake. Final manuscript catalogs additionally use catalog comparisons and
manual decisions, including overrides, which are not performed by this script.

## Conventional seismic stations: Alex's contribution

The station ELEP/PyOcto workflow is not yet included. Please contribute it on a
separate branch and open a pull request into `main` (or use a fork if write access
is unavailable). Suggested location: `ML/stations/`.

Include the actual production scripts/notebooks, dependencies, preprocessing and
picker settings, exact station/channel lists, velocity-model construction, PyOcto
association settings (including minimum picks), study interval, input/output
schemas, and a small run example. Distinguish all-station and five-West runs and
identify which saved outputs belong to each configuration. Do not include tokens,
credentials or large waveform archives. Scientific settings should reflect the
executed workflow, not be inferred from manuscript wording.
