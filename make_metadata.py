import glob
import os
import sys


def main(path, geometry):
    pattern = os.path.join(path, '**/*.png')
    paths = glob.glob(pattern, recursive=True)
    with open(f'train_data/metadata.jsonl', 'w') as fh:
        for path in paths:
            split = path.split('/')
            print(split)
            line = f'{{"file_name": "{split[1]}", "text": "realistic fluid dynamics, flow over {geometry}"}}\n'
            fh.write(line)

if __name__ == '__main__':
    path = sys.argv[1]
    geometry = sys.argv[2]
    main(path, geometry)
