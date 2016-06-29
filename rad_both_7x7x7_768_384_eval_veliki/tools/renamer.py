import os
import sys

for f in os.listdir(sys.argv[1]):
	if f.startswith("b'"):
		os.rename(os.path.join(sys.argv[1], f), os.path.join(sys.argv[1], "{}.bin".format(f[2:-5])))
