# A Yannakakis-rewriting implementation and decision procedure for detecting whether to rewrite or not

## Structure of the repository

The Scala source code for generating the rewriting jar can be found in `scala_query_rewriting_full`.

`benchmark_data` contains the  docker-compose environment. `benchmark_data/benchmark` contains the benchmark code, model training, queries, and benchmark results.

## Running the docker-compose environment

We provide a docker-compose environment with two containers: a PostgreSQL instance into which the benchmark data is loaded. and another container for running the jupyter notebooks.

In `benchmark_data`, run `(sudo) docker-compose up` to start the environment.

## Loading the data

`loading_benchmark_data.ipynb` provides further instructions on how to load the data into PostgreSQL

## Running the benchmarks and training the models

### 0MA queries
1. **Generate the augmented queries** 
   * Run `data_augmentation.ipynb`
   * Output: `scala_commands_augment_filter_agg.txt`
2. **Rewrite the queries**
   * Run the commands using the rewriting jar (rewrite-assembly-0.1.0-SNAPSHOT.jar)
   * Output: `rewritten/{benchmark}_{number}_output.json + rewritten/{benchmark}_{number}_drop.json + rewritten/{benchmark}_{number}_jointree.json`
3. **Run the queries**
   * Run `run_queries_calcite_scala_augment_{DBMS}.ipynb`
   * Output: `results/{DBMS}_scala_comparison_TO_augment.csv`
4. **Extract features**
   * Run `features_{DBMS}.ipynb`
   * Output: `results/features_times_{DBMS}.csv` (also featuresDatabase_{DBMS}.csv, featuresScala_{DBMS}.csv, featuresHypergraph_{DBMS}.csv)
5. **Train ML models**
   * `decision_program_augment_{DBMS}.ipynb`

### Enumeration queries and combination of 0MA+enum
1. **Generated the augmented queries**
   * Run `queries_full_enumeration.ipynb`
   * Output: scala_commands_augment_full_enumeration.txt
2. **Rewrite the queries**
   * Run the commands using the rewriting jar (rewrite-assembly-0.1.0-SNAPSHOT.jar)
   * Output: `output/{benchmark}_{number}_output.json + output/{benchmark}_{number}_drop.json + output/{benchmark}_{number}_jointree.json`
3. **Run the queries**
   * Run `run_queries_calcite_scala_augment_{DBMS}_full_enum_infos.ipynb`
   * Output: `results/{DBMS}_scala_comparison_TO_augment_server_full_enum_infos.csv`
   * Run `run_queries_calcite_scala_augment_{DBMS}_infos.ipynb`
   * Output: `results/{DBMS}_scala_comparison_TO_augment_server_infos.csv`
4. **Extract features**
   * Run `features_{DBMS}_infos.ipynb`
   * Output: `results/features_times_{DBMS}_infos.csv (also featuresDatabase_{DBMS}_infos.csv, featuresScala_{DBMS}_infos.csv)`
5. **Train ML models**
   * Run `decision_tree_{DBMS}_enum.ipynb` (decision tree with plots and results for each setting (enumeration) -> plots in folder plots)
   * Run `decision_tree_{DBMS}_all.ipynb` (decision tree with plots and results for each setting (enumeration+0MA) -> plots in folder plots)
   * Run `decision_tree_{DBMS}_0MA.ipynb` (decision tree with plots and results for each setting (0MA) -> plots in folder plots (to get the same result plots))
6. **Plot stage runtimes**
   * `plot_stages_{DBMS}.ipynb`
   * Output: `plots/{DBMS}_stages_stacked_bar.png, plots/{DBMS}_stages_grouped_bar.png`
