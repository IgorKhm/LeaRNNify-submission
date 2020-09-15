from benchmarking import rand_bench
from benchmarking_no_model_checking import run_extraction_on_dir
from dfa import DFA

if __name__ == '__main__':
    print("Begin")

    # #run_extraction_on_dir
    # check_folder_of_rand("../models/random_bench_10-Jun-2020_06-00-28")
    # check_folder_of_rand("../models/random_bench_03-Jun-2020_05-50-42")
    # complition("../models/random_bench_10-Jun-2020_06-00-28")
    # complition("../models/random_bench_03-Jun-2020_05-50-42")
    # complition("../models/random_bench_21-May-2020_14-40-38")
    # complition("../models/random_bench_21-May-2020_22-02-16/good_ones")
    # run_rand_benchmarks_wo_model_checking(num_of_bench=30)
    # complition("../models/random_bench_21-May-2020_14-40-38")
    # complition("../models/random_bench_21-May-2020_22-02-16/good_ones")

    # run_extraction_on_dir("../models/random_bench_10-Jun-2020_06-00-28")
    # run_extraction_on_dir("../models/random_bench_03-Jun-2020_05-50-42")
    # run_extraction_on_dir("../models/random_bench_21-May-2020_14-40-38")
    # run_extraction_on_dir("../models/random_bench_21-May-2020_22-02-16/good_ones")
    # run_extraction_on_dir("../models/random_bench_23-Jul-2020_07-41-25")

    rand_bench(timeout=100, check_flows=True)
