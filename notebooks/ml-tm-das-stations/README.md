# ML-TM-DAS-STATIONS

Reproduce the catalog comparisons between DAS, the five nearby seismic stations
(KAUT, LON, MILK, PARA and RER), and the full Rainier station network. Both
notebooks start from saved event/detection CSVs. They do not run phase picking,
association, waveform processing, or a new manual review.

## Notebooks

- **`Rainier_summary_ComCat_TM_ML.ipynb`** generates the current three-panel
  summary: (a) local ComCat recovery by TM and ML, (b) direct TM versus ML, and
  (c) direct DAS versus five-station ML. Use this notebook to edit the paper figure.
- **`Rainier_DAS_vs_stations_GaMMA.ipynb`** contains the complete comparison,
  including catalog accounting, time differences, tolerance sensitivity,
  monthly counts, regional ComCat comparisons and local-reference diagnostics.

The figure is displayed inline when the summary notebook runs. Its full caption
is in [figure_caption.txt](figure_caption.txt).

## Run

From the repository root:

```bash
python -m pip install -r notebooks/ml-tm-das-stations/requirements.txt
python notebooks/ml-tm-das-stations/prepare_inputs.py --input-dir /path/to/local/inputs
python -m jupyter lab notebooks/ml-tm-das-stations/Rainier_summary_ComCat_TM_ML.ipynb
```

Research CSVs are not published in this pull request. The setup helper copies
the eight files from the existing local notebook bundle; see the required
[input filenames](../../text-file/ml-tm-das-stations/inputs/README.md).
Select a Python kernel and choose **Run All**. The notebooks locate the local
inputs from either the repository root or the notebook directory. Once inputs
are prepared, no workstation-specific paths or server access are required.
The first configuration cell controls tolerance, dates, ComCat
radius and DAS review selection. Section 8 of the summary notebook controls
colors, labels and panel layout.

Outputs are written to separate, git-ignored `results/summary/` and
`results/comprehensive/` folders beside the notebooks. Exports include PNG, PDF,
SVG, calculated tables and individual matched pairs. Notebook outputs are
cleared in Git so changes remain reviewable.

## Default analysis

- Common UTC interval: 23 August 2023 21:00 inclusive to 22 November 2023 00:00
  exclusive. Common dates do not establish identical data availability.
- GaMMA is the ML associator. Current DAS results use PhaseNet-DAS picks;
  station picks are described as ELEP by the workflow owner. The notebooks do
  not establish or rerun the upstream picker configuration.
- Matching is one-to-one within **±7 seconds**, maximizing pair count and then
  minimizing summed absolute time differences. A separate bipartite matching
  calculation checks maximum cardinality. Each catalog comparison is independent.
- Panel (a) uses **216 ComCat earthquakes within 25 km** of the original query
  center (46.879967 N, 121.726906 W). Selection uses ComCat positions, without
  also restricting ML positions. Panels (b) and (c) use complete selected
  catalogs, with no ComCat or epicentral-radius filter.
- DAS TM uses strict **CC > 0.07**. Station TM retains all supplied detections.
- The regional ComCat input is a cached download, not a fresh query. Earthquake
  types are selected within 45–48 N and 123–120 W; no depth filter is imposed
  by default.

## Expected results at the defaults

| Selected catalog | Records | Local ComCat recovered |
| --- | ---: | ---: |
| DAS ML | 478 | 152 / 216 |
| Five-station ML | 266 | 124 / 216 |
| Full-network ML | 1,383 | 208 / 216 |
| DAS TM | 85 | 53 / 216 |
| Station TM | 92 | 62 / 216 |

Panel (b): DAS has 39 TM-only records, 46 shared pairs and 432 ML-only records.
Stations have 55 TM-only records, 37 shared pairs and 229 ML-only records.
Each union counts a shared pair once.

Panel (c): DAS and five-station ML share 117 pairs, with 361 DAS-only and 149
five-station-only records. Shared pairs appear in both input-catalog bars and
must be counted once when combining the catalogs.

## Interpretation and provenance

The raw DAS GaMMA run has 942 candidates. A previous visual review consolidated
483 confirmed groups; the saved correspondence links 478 groups to distinct
updated GaMMA records, including 138 uncertain links retained by default. Five
unlinked groups are omitted. This correspondence is not a new visual review.
The 266 five-station records are accepted as valid for the study comparison.
The raw full-network CSV has 1,384 rows; one is outside the configured interval.

Station-TM production stations are unverified, so panel (b) is not a controlled
five-station TM versus five-station ML experiment. TM stores correlation-peak
detection times while ML stores associated origin times. The peak-time reference
has not been converted, and DAS clock drift has not been corrected. A temporal
nonmatch does not establish a distinct or previously uncatalogued earthquake.
The comparison does not measure location accuracy or false-positive rates.

Input filenames and SHA-256 hashes are calculated during notebook execution.
The setup helper also preserves `source_manifest.json` locally when available.
The required local data are event tables, review correspondence and a cached
reference catalog. Waveforms and individual phase-pick archives are not required.
