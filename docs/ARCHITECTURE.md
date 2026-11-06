# Architecture

pipeline.py copies src and vendor into each binary run. The flat layout preserves configuration paths, artifact checks and worker imports.

Binary detection uses a masked contextual VAE, background reconstruction evidence, reward evidence, bounded categorical-tail correction and a hierarchical policy. The evidence modules are components of one detector.
