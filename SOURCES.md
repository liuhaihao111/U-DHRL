# Third-party source dependencies

- `vendor/imperial-qore__TranAD/src/dlutils.py` is copied from `imperial-qore/TranAD` at commit `7ffb98d0c18189cc3d9ab732b4cb0278200a0af0`. Its upstream BSD 3-Clause license is included as `vendor/imperial-qore__TranAD/LICENSE`.
- GRASPED is referenced by the baseline implementation but is not bundled because the upstream repository at `guanwei49/GRASPED` provides no license file or reuse terms. Obtain `model/GRU_AE.py` from commit `14fb81df161a952067e13154c4148af87b969b14` only after confirming permission to use and redistribute it. Expected local path: `vendor/guanwei49__GRASPED/model/GRU_AE.py`.

The TranAD implementation is used for the paper's baseline. GRASPED source and its licensing status should be resolved before publishing a runnable repository that includes that baseline.
