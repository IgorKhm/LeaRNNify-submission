from benchmarking import rand_pregenerated_benchmarks
from benchmarking_no_model_checking import run_extraction_on_dir
from dfa import DFA

if __name__ == '__main__':
    rand_pregenerated_benchmarks(timeout=100, check_flows=True)
