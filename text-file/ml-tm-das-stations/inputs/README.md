# Local research inputs

Research CSVs are not published with this pull request. This directory is
git-ignored except for this README and `.gitignore`.

From the repository root, copy the eight inputs from the existing local notebook
bundle with:

```bash
python notebooks/ml-tm-das-stations/prepare_inputs.py --input-dir /path/to/local/inputs
```

Required files:

| File | Default raw rows |
| --- | ---: |
| `gamma_das_candidates.csv` | 942 |
| `gamma_five_station_events.csv` | 266 |
| `gamma_all_station_events.csv` | 1,384 |
| `tm_das_detections.csv` | 148 |
| `tm_station_detections.csv` | 92 |
| `gamma_original_confirmed_groups.csv` | 483 |
| `gamma_confirmed_group_to_new_run.csv` | 483 |
| `comcat_regional_raw.csv` | 1,216 |

The helper preserves each input's bytes, including line endings, and copies the
optional local `source_manifest.json`. Notebook execution recalculates SHA-256
hashes and displays source schemas and filtering steps. A replacement association
catalog may require updating the historical-review correspondence.
