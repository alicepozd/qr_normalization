import sys
from source.get_unddeformed_images import *

source_path = sys.argv[1]
dataset_size, no_norm_stat, norm_stat, dataset_size, bad_cases = get_statistics(source_path)

