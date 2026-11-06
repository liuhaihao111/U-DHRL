# Dependencies

Use Python 3.10 with requirements.txt for a fresh environment. PyTorch, NumPy, scikit-learn and SciPy cover the direct numerical imports. XES parsing uses the standard library. CUDA is optional.

The saved paper environment reports torch 2.3.1+cu121, NumPy 1.24.4 and scikit-learn 1.2.1. Frozen-model regression uses that environment. The package dependency versions are specified in requirements.txt.

TranAD uses the bundled BSD helper. GRASPED is optional: run `python setup_grasped.py` to obtain the pinned upstream implementation before requesting that baseline. See THIRD_PARTY.md.
