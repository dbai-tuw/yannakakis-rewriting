# Decision program if rewriting a SQL query is useful

In `benchmark_data` the code for loading the benchmarks and queries, as well as code for running the queries can be found.  

1. loading_benchmark_data.ipynb -> data set loading and postgresql + duckDB user creating
2. data_augmentation.ipynb -> get the augmented data -> scala_commands_augment_filter_agg.txt
3. These commands into the Scala programm, get the rewritten queries -> rewritten/{benchmark}_{number}_output.json + rewritten/{benchmark}_{number}_drop.json + rewritten/{benchmark}_{number}_jointree.json (rewritten folder not complete...would be 8800 files)
4. run_queries_calcite_scala_augment_{DBMS}.ipynb -> running queries in both versions for one DBMS -> results/{DBMS}_scala_comparison_TO_augment.csv
5. features.ipynb -> get all features in a dataframe together with the times -> results/features_times.csv
6. decision_program_augment_{DBMS}.ipynb -> ML models using the times to get the decision program



In `scala` the Scala code is saved.
