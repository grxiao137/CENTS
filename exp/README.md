# exp

This directory contains experiment results and scripts for table understanding tasks using the CENTS framework.

We have prepared our own experiment generation results under each `out` folder under each model & budget.
For each task, you can run `run_eval_all.sh` to quickly reproduce our results.
If you wish to run experiments from scratch, please find `run_cta.sh` under `CTA` folder, `run_re.sh` under the `RE` folder, and `run_sa.sh` under the `SA` folder to regenerate all output under each `out` folder.

## Structure

```

.
├── README.md
├── setup.py
├── exp
│   ├── requirements.txt
│   ├── run_eval_all.sh
│   ├── CTA
│   │   ├── cta_eval.py
│   │   ├── run_cta.sh // all runtime 
│   │   ├── run_eval_all.sh // all micro f1
│   │   ├── raw
│   │   └── (models & budgets)
│   ├── RE
│   │   ├── re_eval.py
│   │   ├── run_re.sh // all runtime
│   │   ├── run_eval_all.sh // all micro f1
│   │   ├── raw
│   │   └── (models & budgets)
│   └── SA
│       ├── sa_eval.py
│       ├── run_sa.sh
│       ├── run_eval_all.sh // all micro f1
│       ├── raw
│       └── (models & budgets)

```

Top level is three table understanding tasks, and then under each folder, we have `run_xx.sh` to run each experiments end-to-end, and `run_eval_all.sh` to evaluate all generation output end-to-end. If you run the end-to-end pipeline, it will output to the `out` folder under each sub-dir so make sure you have backup for that folder everytime.

## Instructions
To re-run end-to-end results, make sure you have done the following:

- Follow the instruction [here](../src/data/README.md) to download all datasets, save each corresponding raw tables under each `raw` folder under each task. Specifically, put `CTA/CTA_Test` from google drive under `exp/CTA/raw`, `RE/RE_Test` from google drive under `exp/RE/raw`, the three json files under `TURL` under `exp/SA/raw`.
- Make sure you have your open-ai API key set-up as well as your MOSEK solver liscense set up. For setting up MOSEK, please follow instructiosn on their [website](https://www.mosek.com/).
- For OpenAI key, make sure you have done setting up your api key via ```export OPENAI_API_KEY="sk-xxxxxx"```
- Make sure your machine has at least one **NVDIA L40S** GPU (or a better one if you are richer than me ...) on your machine.
- For SA, the `run.sh` by default uses top-k=50, you can change it to 10, 50, 100, and 1000 inside the script as in the experiments.
- For CTA/RE runtime, you should re-run the entire pipeline to get the results for CENTS runtime and LLM runtime.



### Have you set up the dataset correctly?

For CTA you should see

```
exp/CTA/raw
└── CTA_Test
    ├── CTA_test_corner_cases_gt.csv
    ├── CTA_test_format_heterogeneity_gt.csv
    ├── CTA_test_missing_values_gt.csv
    ├── CTA_test_random_gt.csv
    └── Test/
```

For RE you should see


```
exp/RE/raw
└── RE_Test
    ├── CPA_test_gt.csv
    ├── CPA_test_random_gt.csv
    └── Test/
```

For SA you should see

```
exp/SA/raw
├── header_vocab.txt
├── test_headers.json
└── test_tables.jsonl
```



### Example - Evaluate all generation results of CTA

```bash
cd ./CTA
bash run_eval_all.sh 
```

This script will invoke the `cta_eval.py` to evaluate against all generation results to gen scores for all models (i.e., gpt3.5/gpt4o/deepseek/tablegptv2) under all budgets.


(***NOTE***: you do NOT need to re-run the entire pipelines to run this, we have prepared our results under each `out` folder under each model and budgets. For example, `exp/CTA/gpt3.5/out/2000` contains all json files from our experiments.)


### Example - Run full set of experiments of CTA

```bash
cd ./CTA
bash run_cta.sh
```

This script will invoke the `main.py` file under each folder to gen outputs json for all models (i.e., gpt3.5/gpt4o/deepseek/tablegptv2) under all budgets.

During execution, each `main.py` will create a folder under `out/` that contains `shard.json` files, each `shard.json` file should have the following structure:
```
{
    "gt": {
        "CreativeWork_squirrellyminds.com_September2020_CTA.json.gz": [...],
        ...
    },
   "pred": {
        "CreativeWork_squirrellyminds.com_September2020_CTA.json.gz": [...],
        ...
    },
}
```
where the length of gt for each table id should be match that of the id under the pred, then please run

```bash
bash run_eval_all.sh
```

which compares the gt against the pred and calculate the micro f1.

(***NOTE***: For OpenAI experiments, make sure you understand that running the experiements will cost you money b/c they charge by per-token [price](https://openai.com/api/pricing/))





