'''

Used in combination with validation_set_picker to copy appropriate label and potentials files

Script arguments:
    1: file containing file names to be copied (without extension)
    2: source directory
    3: destination directory
    4: file extension


'''

import sys
import os
from shutil import copyfile

if not os.path.exists(sys.argv[3]):
    os.makedirs(sys.argv[3])

with open(sys.argv[1], "r") as f:
    for file in f:
        file = file.strip()
        copyfile(os.path.join(sys.argv[2], "{}.{}".format(file, sys.argv[4])),
                 os.path.join(sys.argv[3], "{}.{}".format(file, sys.argv[4])))
