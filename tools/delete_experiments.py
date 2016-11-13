import os
import shutil
from argparse import ArgumentParser


def main(filename, header_flag, delete_logfile):
    if not os.path.exists(filename):
        print 'Unexistent file: {}, Bye!'.format(filename)
        return None

    exp_root = os.path.dirname(os.path.abspath(filename))
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if header_flag and i == 0:
                continue
            exp_id = line.rstrip('\n')

            exp_dir = os.path.join(exp_root, exp_id)
            logfile = os.path.join(exp_root, 'log', exp_id + '.out')
            if os.path.isdir(exp_dir):
                shutil.rmtree(exp_dir)

            if delete_logfile and os.path.isfile(logfile):
                os.remove(logfile)


if __name__ == '__main__':
    description = (
        'Delete "non-worth" data of experiments from a root folder.\n'
        'Each experiment is labeled with an "exp-id". Its data is located '
        'inside a folder in the first level of the root folder. Additionally, '
        'it can delete an extra logfile for each exp-id assuming that is '
        'located in (root/log/exp-id.out) ')
    p = ArgumentParser(description=description)
    p.add_argument('-f', '--filename', default='non-existent',
                   help='Fullpath CSV-file with experiment-ids to delete')
    p.add_argument('-hf', '--header-flag', action='store_true',
                   help='Pass it if CSV has a header')
    p.add_argument('-dl', '--delete-logfile', action='store_false',
                   help='Pass it to avoid deleting logfiles')
    main(**vars(p.parse_args()))
