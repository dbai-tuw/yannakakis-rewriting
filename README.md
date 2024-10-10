# Decision program if rewriting a SQL query is useful

In `benchmark_data` the code for loading the benchmarks and queries, as well as code for running the queries can be found.  

## Loading the data
loading_benchmark_data.ipynb -> data set loading and postgresql + duckDB user creating

## 0MA queries
1. data_augmentation.ipynb -> get the augmented data -> scala_commands_augment_filter_agg.txt
2. These commands into the Scala programm, get the rewritten queries -> rewritten/{benchmark}_{number}_output.json + rewritten/{benchmark}_{number}_drop.json + rewritten/{benchmark}_{number}_jointree.json (rewritten folder not complete...would be 8800 files)
3. run_queries_calcite_scala_augment_{DBMS}.ipynb -> running queries in both versions for one DBMS -> results/{DBMS}_scala_comparison_TO_augment.csv
4. features_{DBMS}.ipynb -> get all features in a dataframe together with the times -> results/features_times_{DBMS}.csv (also featuresDatabase_{DBMS}.csv, featuresScala_{DBMS}.csv, featuresHypergraph_{DBMS}.csv)
5. decision_program_augment_{DBMS}.ipynb -> ML models using the times to get the decision program

## enum queries and combination of 0MA+enum
1. queries_full_enumeration.ipynb -> get the (augmented) enumeration queries -> scala_commands_augment_full_enumeration.txt
                                  -> do the rewriting in Python using the jar file (rewrite-assembly-0.1.0-SNAPSHOT.jar), which produces output/{benchmark}_{number}_output.json + output/{benchmark}_{number}_drop.json + output/{benchmark}_{number}_jointree.json (output folder not complete)
2. run_queries_calcite_scala_augment_{DBMS}_full_enum_infos.ipynb -> running queries in both versions for one DBMS and get the infos per stage additionally -> results/{DBMS}_scala_comparison_TO_augment_server_full_enum_infos.csv
   run_queries_calcite_scala_augment_{DBMS}_infos.ipynb -> running queries in both versions for one DBMS and get the infos per stage additionally -> results/{DBMS}_scala_comparison_TO_augment_server_infos.csv
3. features_{DBMS}_infos.ipynb -> get all features in a dataframe together with the times -> results/features_times_{DBMS}_infos.csv (also featuresDatabase_{DBMS}_infos.csv, featuresScala_{DBMS}_infos.csv)
4. decision_tree_{DBMS}_enum.ipynb -> decision tree with plots and results for each setting (enumeration) -> plots in folder plots
   decision_tree_{DBMS}_all.ipynb -> decision tree with plots and results for each setting (enumeration+0MA) -> plots in folder plots
   decision_tree_{DBMS}_0MA.ipynb -> decision tree with plots and results for each setting (0MA) -> plots in folder plots (to get the same result plots)
6. plot_stages_{DBMS}.ipynb -> produces the plots for the runtimes per stages -> plots/{DBMS}_stages_stacked_bar.png, plots/{DBMS}_stages_grouped_bar.png

## Scala
In `scala_query_rewriting_full` the Scala code is saved, which works for 0MA and enumeration queries and produces a jar file.
