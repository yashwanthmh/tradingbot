"""The registry: which strategies exist, where they came from, what they may do.

`lineage` holds identity and ancestry, `promotion` holds the gate, `ladder`
holds the size ratchet. Split that way because the three change for different
reasons: identity is bookkeeping, the gate is the safety argument, and the
ladder is the operating procedure. `model_store` is the same bookkeeping for
the models a spec may read: artifacts by hash, admitted by the ledger.
"""
