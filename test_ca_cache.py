import sys
sys.path.append(".")
from build_adjusted_master import main

# Mock testing just one symbol to force the CA block to run fast
main(target_symbol='ANGELONE')
