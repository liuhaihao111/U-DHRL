# Validation performed

- Python syntax checks for all packaged modules.
- Synthetic XES log: 160 cases, joint activity/lifecycle encoding, disjoint 70/5/5/20 partitions, all 11 injection rates checked.
- End-to-end CPU smoke run of U-DHRL and Naive from raw synthetic XES through training, checkpoint reload, frozen evaluation and summary export; completed successfully.
- Validation environment: Python 3.12, PyTorch 2.3.1+cpu, NumPy 1.26.4, scikit-learn 1.5.2.
- The synthetic data and results are not included. Reproduce the functional check with smoke_test.py.
- No complete nine-log/five-seed training or paper-number reproduction was performed.

## Source changes

All inherited source is from the specified author ZIP. No other local historical source is included. pipeline.py, smoke_test.py, benchmark.py, setup_grasped.py and src/standalone_data.py are new integration utilities. Model/reward implementations are unchanged. src/common.py only adds null values for undefined ROC-AUC/no-positive average precision in tiny single-class test sets; ordinary two-class calculations are unchanged. config.json removes unusable historical directory references; numeric training settings are retained except when --smoke is explicitly requested.

The release uses newly generated deterministic splits/recipes; no claim that these match all historical server artifacts is made. Later V30/V41 refinement and the complete paper's ablation/sensitivity experiments are outside this core source release.

On-demand inference was exercised using the smoke-run models. Eager and on-demand binary decisions and class outputs matched. This is a functional check, not a timing reproduction.
