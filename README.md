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
   * Run the commands using the rewriting jar
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
1. queries_full_enumeration.ipynb -> get the (augmented) enumeration queries -> scala_commands_augment_full_enumeration.txt
                                  -> do the rewriting in Python using the jar file (rewrite-assembly-0.1.0-SNAPSHOT.jar), which produces output/{benchmark}_{number}_output.json + output/{benchmark}_{number}_drop.json + output/{benchmark}_{number}_jointree.json (output folder not complete)
2. run_queries_calcite_scala_augment_{DBMS}_full_enum_infos.ipynb -> running queries in both versions for one DBMS and get the infos per stage additionally -> results/{DBMS}_scala_comparison_TO_augment_server_full_enum_infos.csv
   run_queries_calcite_scala_augment_{DBMS}_infos.ipynb -> running queries in both versions for one DBMS and get the infos per stage additionally -> results/{DBMS}_scala_comparison_TO_augment_server_infos.csv
3. features_{DBMS}_infos.ipynb -> get all features in a dataframe together with the times -> results/features_times_{DBMS}_infos.csv (also featuresDatabase_{DBMS}_infos.csv, featuresScala_{DBMS}_infos.csv)
4. decision_tree_{DBMS}_enum.ipynb -> decision tree with plots and results for each setting (enumeration) -> plots in folder plots
   decision_tree_{DBMS}_all.ipynb -> decision tree with plots and results for each setting (enumeration+0MA) -> plots in folder plots
   decision_tree_{DBMS}_0MA.ipynb -> decision tree with plots and results for each setting (0MA) -> plots in folder plots (to get the same result plots)
6. plot_stages_{DBMS}.ipynb -> produces the plots for the runtimes per stages -> plots/{DBMS}_stages_stacked_bar.png, plots/{DBMS}_stages_grouped_bar.png
