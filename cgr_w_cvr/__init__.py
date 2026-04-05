"""CGR with CVR: profit-aware reference code for ad-feed generative reranking.

Inspired by arXiv:2603.04227 (Bilibili, 2026).

This package starts from the paper-inspired CGR implementation and extends it
with a third CVR task plus an organic-profit reward term. It intentionally
keeps several components simplified and should not be read as an exact
reproduction of every modeling or decoding detail.
"""
