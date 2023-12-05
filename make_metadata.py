import glob
import os
import sys


def main(path):
    pattern = os.path.join(path, '**/*.png')
    paths = glob.glob(pattern, recursive=True)
    name = os.path.split(path)[0]
    with open(f'{name}_metadata.json', 'w') as fh:
        for path in paths:
            print(path.split('/'))
            name, res, var = path.split('/')[:3]
            line = f'{{"file_name": "{path}", "text": "flow over {res} {name}"}}\n'
            fh.write(line)
    #    pass

if __name__ == '__main__':
    path = sys.argv[1]
    main(path)
