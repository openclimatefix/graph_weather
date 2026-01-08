"""
AI-based Data Assimilation Package

Implements AI-based data assimilation following the approach described in:
"AI-Based Data Assimilation: Learning the Functional of Analysis Estimation" (arXiv:2406.00390)

This package provides neural networks that learn to produce optimal analysis states
by minimizing the 3D-Var cost function in a self-supervised manner, without requiring
ground-truth labels.
"""