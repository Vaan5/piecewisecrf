'''

Picks a subset of the validation set for grid_search

Script arguments:
    1: directory containing validation ppm images
    2: output directory
    3: how many images will be picked


'''
import random
import os
import sys
from shutil import copyfile

all_files = os.listdir(sys.argv[1])
all_files = [x[0:-4] for x in all_files]

picked_files = random.sample(all_files, int(sys.argv[3]))

for file in picked_files:
    print(file)
    copyfile(os.path.join(sys.argv[1], "{}.ppm".format(file)),
             os.path.join(sys.argv[2], "{}.ppm".format(file)))    # input
