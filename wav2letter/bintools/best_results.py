# examples:
# ~/anaconda/bin/python best_results.py -10 # last ten days
# ~/anaconda/bin/python best_results.py +100 filter # last 100 experiment filtered for valid < 25
# ~/anaconda/bin/python best_results.py list_of_folders.txt # produced as you want
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
usage = """
~/anaconda/bin/python best_results.py arg [filter]
arg in:
    file: one xp per line
    N last days: -N
    N most recent: +N
"""
import sys
if len(sys.argv) < 2 or '-h' in sys.argv[1]:
    print("usage:" + usage)
    sys.exit(-1)

basedir = '/mnt/vol/gfsai-east/ai-group/teams/wav2letter/experiments/'
from subprocess import check_output
l = []
if sys.argv[1][0] == '-':
    cmd = "find " + basedir + " -maxdepth 1 -type d -mtime -" + sys.argv[1][1:]
    print(cmd)
    l = map(lambda x: x.split('/')[-1], check_output(cmd, shell=True).split('\n'))
elif sys.argv[1][0] == '+':
    cmd = "ls -lth " + basedir + " | head -n " + sys.argv[1][1:] + " | sed -e 's/  / /g' | cut -d' ' -f10"
    print(cmd)
    l = check_output(cmd, shell=True).split('\n')
else:
    with open(sys.argv[1]) as rf:
        # e.g. sys.argv[1] can be obtained by
        # find /mnt/vol/gfsai-east/ai-group/teams/wav2letter/experiments/ -maxdepth 1 -type d -mtime -10
        l = map(lambda x: x.rstrip('\n').split('/')[-1], rf.readlines())

l = filter(lambda x: len(x), l)

for h in l:
    try:
        logfile = basedir + h + '/log'
        with open(logfile) as rf:
            min_valid = 10000
            dataset = ''
            arch = ''
            feats = ''
            for line in rf:
                if 'config ' in line:
                    dataset = line.rstrip('\n').split()[1]
                if 'mfsc ' in line and 'true' in line:
                    feats = 'mfsc' + feats
                if 'mfcc ' in line and 'true' in line:
                    feats = 'mfcc' + feats
                if 'delta_size 9' in line:
                    feats = feats + '_deltas'
                if 'arch ' in line and 'archdir' not in line:
                    arch = line.split(' ')[1].rstrip('\n')

                sp = ''
                if 'valid LER ' in line:
                    sp = 'valid LER '
                elif 'nov93dev LER ' in line:
                    sp = 'nov93dev LER '
                elif 'dev-clean LER ' in line:
                    sp = 'dev-clean LER '

                if sp != '':
                    ler = line.split(sp)[1].split(' ')[0]
                    ler = float(ler)
                    if ler < min_valid:
                        min_valid = ler
            if not(len(sys.argv) > 2 and sys.argv[2] == 'filter')\
                    or min_valid < 25:
                print(dataset, arch, feats, min_valid, logfile)
    except Exception as e:
        print(e)
