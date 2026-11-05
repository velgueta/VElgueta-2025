"""Copy local research CSVs into the git-ignored notebook input directory."""
from pathlib import Path
import argparse
import shutil

FILES = (
    'gamma_das_candidates.csv', 'gamma_five_station_events.csv',
    'gamma_all_station_events.csv', 'tm_das_detections.csv',
    'tm_station_detections.csv', 'gamma_original_confirmed_groups.csv',
    'gamma_confirmed_group_to_new_run.csv', 'comcat_regional_raw.csv',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Existing local folder with the eight input CSVs')
    args = parser.parse_args()
    source = args.input_dir.expanduser().resolve()
    target = Path(__file__).resolve().parents[2] / 'text-file/ml-tm-das-stations/inputs'
    missing = [name for name in FILES if not (source/name).is_file()]
    if missing:
        parser.error('Missing input CSVs: '+', '.join(missing))
    names = list(FILES)
    if (source/'source_manifest.json').is_file():
        names.append('source_manifest.json')
    # Check every existing destination before copying any file.
    conflicts = [name for name in names if (target/name).exists()
                 and (target/name).read_bytes() != (source/name).read_bytes()]
    if conflicts:
        parser.error('Existing local files differ; move them before replacing: '+', '.join(conflicts))
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        if source/name != target/name:
            shutil.copyfile(source/name, target/name)
    print(f'Prepared {len(FILES)} local CSVs in {target}; research inputs remain git-ignored.')


if __name__ == '__main__':
    main()
