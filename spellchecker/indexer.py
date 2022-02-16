import numpy as np
import re
import sys
import pickle
from trie import *
from lev import *
from error_model import *
from language_model import *

if __name__ == '__main__':
    print('author is Ivan Bystrov', file=sys.stderr)
    print('doesnt pickle anything', file=sys.stderr)
    print('cause dont want to break the system with no space left on device error', file=sys.stderr)
    print('so, build model directly in checker', file=sys.stderr)
    print('also, added timestamp to separate fixing process from building models process', file=sys.stderr)
