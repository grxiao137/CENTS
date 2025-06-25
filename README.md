# CENTS

Prototype implementation of **CENTS**, that consists of supplementary materials and implementations.



---

## Repo Structure

```

.
├── README.md
├── setup.py
├── exp
│   ├── requirements.txt
│   ├── run_eval_all.sh
│   ├── CTA
│   │   ├── cta_eval.py
│   │   ├── run_cta.sh
│   │   ├── run_eval_all.sh
│   │   ├── raw
│   │   └── gpt3.5/ gpt4o/ deepseek/ tabgptv2/
│   ├── RE
│   │   ├── re_eval.py
│   │   ├── run_re.sh
│   │   ├── run_eval_all.sh
│   │   ├── raw
│   │   └── (model sub-dirs similar to CTA)
│   └── SA
│       ├── sa_eval.py
│       ├── run_sa.sh
│       ├── run_eval_all.sh
│       ├── raw
│       └── (model sub-dirs similar to CTA)
├── src
│   ├── utils
│   ├── cdr
│   │   ├── message.py
│   │   ├── scorer/
│   │   └── solver/
│   ├── connector
│   │   ├── base_connector.py
│   │   └── gpt_connector.py
│   ├── data
│   │   ├── const.py
│   │   ├── map1749.txt
│   │   ├── topk-turl.pkl
│   │   ├── cc.en.100.bin
│   │   └── README.md
│   ├── serializer
│   │   ├── base_ser.py
│   │   └── dfser.py
│   └── tdr
│       ├── dataset.py
│       ├── gentopk_sa.py
│       ├── model.py
│       └── train.py

````

### `src/` 
`src/` folder contains the prototype implementation of CENTS. It generally contains the following folders:
* **utils/** - a folder contaning generic helper functions
* **cdr/** – context-data-reduction logic (`scorer/` for scoring cells & `solver/` for picking).  
* **connector/** – thin wrappers around external LLM endpoints (OpenAI, etc.).  Note that as of now, we only have OpenAI wrapperr available.
* **data/** – static artefacts such as FastText vectors and constants vals.
* **serializer/** – utilities for turning tables into NL strings. Note that here we use dataframe style serializer across.
* **tdr/** – task-data-reduction, this is built on top of [DODUO](https://github.com/megagonlabs/doduo). Please follow its own instruction for installation first before running tdr. It is a two-staged framework where tdr will first generate a top-k pkl file for TURL, and then durign inference we uses that top-k pkl file to retrieve possible label space for each table. For your convenience, we have provided our tdr-ed pkl file under `src/data/topk-turl.pkl`


### `exp/`
`src/` folder contains the evaluation results & scripts for CENTS. It generally contains the following folders:

* **CTA/** – Exp results & scripts for CTA task with different models and budget.
* **RE/** –  Exp results & scripts for RE task with different models and budget.
* **SA/** –  Exp results & scripts for SA task with different models and budget.
* **run_eval_all.sh/** – Eval script over all experiments generation results. 

Please refer to `exp/README.md` for detailed instructions

---

## Install

(***NOTE***: you MUST have a GPU available for installing torch GPU ver in requirements.txt, but if you only want to try GPT series, comment out the last few lines in `exp/requirements.txt` that are torch related.)


```bash
git clone https://github.com/grxiao137/CENTS.git && cd CENTS

conda create -n cents python=3.12
conda activate cents

pip install -e .
pip install -r exp/requirements.txt
````

---

## Running the Benchmarks

1. Download all benchmark datasets (see `src/data/README.md`).
2. Follow the detailed task instructions in `exp/README.md`.

### Quick evaluation over every generated result (all three tasks)

```bash
cd exp
bash run_eval_all.sh
```

---

Questions? Email grxiao @ cs dot washington dot edu
