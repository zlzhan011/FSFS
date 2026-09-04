# Fairness-Aware Streaming Feature Selection with Causal Graphs

## Overview

This repository provides the implementation of a fairness-aware streaming
feature-selection framework based on causal graphs.

The method is designed to identify useful features in evolving data streams while
reducing the influence of features that may transmit sensitive-attribute information
and contribute to discriminatory predictions.

## Main Functionality

The repository includes:

- fairness-aware streaming feature selection;
- causal-graph-based feature analysis;
- correlation and discrimination measurement;
- classification and learning modules;
- statistical comparison tools;
- experiments on Adult, COMPAS, German Credit, Communities and Crime,
  and credit-card datasets.

## Motivation

Machine-learning systems may preserve or amplify algorithmic bias when proxy
features indirectly encode sensitive information. This project investigates
feature-selection strategies that preserve predictive utility while reducing
potential sources of unfairness in data-driven decision systems.

## Repository Structure

- `analysis_adult/`
- `analysis_compas/`
- `analysis_german/`
- `analysis_communities/`
- `analysis_credit_card/`
- `correlation_measure/`
- `discrimination/`
- `learning_module/`
- `statistical_comparison/`

## Related Publication

"Fairness-Aware Streaming Feature Selection with Causal Graphs,"
IEEE International Conference on Systems, Man, and Cybernetics, 2024.

[完整 citation]
