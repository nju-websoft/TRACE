"""Build the arrow training manifests expected by the released grid launchers."""
import argparse
from pathlib import Path
from gen_data.format_data import build_manifest, convert_to_arrow_format

DATASETS = {
    'cbd': [('cbd', 'cbd_arrow_training_all.json'), ('cbd_val', 'cbd_val_arrow_training_all.json')],
    'fca': [('fca', 'fca_arrow_training_all.json')],
    'fcb': [('fcb', 'fcb_arrow_training_all.json'), ('fcb_val', 'fcb_val_arrow_training_all.json')],
    'flowlearn': [('flowlearn', 'flowlearn_arrow_training_all.json')],
    'flowvqa': [('flowvqa', 'flowvqa_arrow_training_all.json')],
    'bpmn': [('bpmn', 'bpmn_arrow_training_all.json'), ('bpmn_dev', 'bpmn_val_arrow_training_all.json')],
    **{f'flowgen_{difficulty}': [(f'flowgen_train_{difficulty}', f'flowgen_train_{difficulty}.json')]
       for difficulty in ('easy', 'medium', 'hard')},
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('Dataset/data_4_training'))
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    for dataset in args.datasets:
        for folder, filename in DATASETS[dataset]:
            source = args.data_dir / folder
            if not source.is_dir():
                raise SystemExit(f'Missing intermediate dataset directory: {source}')
            manifest = build_manifest(source)
            if not manifest:
                raise SystemExit(f'No usable annotations in {source}')
            convert_to_arrow_format(manifest, args.data_dir / filename, dataset)

if __name__ == '__main__':
    main()
