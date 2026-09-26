"""The ML signal layer: models whose score is one more term a strategy can read.

A model here is not a strategy. It is a scorer inside the one feature pipeline,
pinned by the hash of its artifact, and a DSL spec decides on its score the way
it decides on a moving average. Everything downstream — trials, the sealed
holdout, the promotion gate, the allocator, the loop, `tb replay` — therefore
treats an ML strategy exactly as it treats any other, and none of it needed a
second code path to do so.
"""
