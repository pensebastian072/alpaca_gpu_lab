# Results log

Every gate verdict, honest FAILs included — that is the system working, not a bug.

| date | battery | horizon | n | PF | Sharpe | DSR ratio | PBO | n_trials | verdict | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-17 | B01_baselines_core7 | 5 | 35 | 0.660 | -0.179 | -0.970 | - | 1 | FAIL | logreg+rf {'model': 'logreg', 'C': 1.0, 'max_iter': 2000, 'class_weight': 'balanced'} thr=0.6 |
| 2026-07-17 | B01_baselines_core7 | 15 | 16 | 0.746 | -0.123 | -2.372 | - | 2 | FAIL | logreg+rf {'model': 'logreg', 'C': 1.0, 'max_iter': 2000, 'class_weight': 'balanced'} thr=0.6 |
| 2026-07-17 | B01_baselines_core7 | 30 | 14 | 0.388 | -0.403 | -3.695 | - | 3 | FAIL | logreg+rf {'model': 'logreg', 'C': 1.0, 'max_iter': 2000, 'class_weight': 'balanced'} thr=0.6 |
| 2026-07-17 | B01_baselines_core7 | 5 | 13 | 0.321 | -0.491 | -4.740 | - | 4 | FAIL | logreg+rf {'model': 'rf', 'n_estimators': 300, 'max_depth': 6, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1} thr=0.6 |
| 2026-07-17 | B01_baselines_core7 | 5 | 35 | 0.660 | -0.179 | -8.479 | - | 7 | FAIL | logreg+rf {'model': 'logreg', 'C': 1.0, 'max_iter': 2000, 'class_weight': 'balanced'} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B01_baselines_core7 | 15 | 16 | 0.746 | -0.123 | -6.068 | - | 9 | FAIL | logreg+rf {'model': 'logreg', 'C': 1.0, 'max_iter': 2000, 'class_weight': 'balanced'} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B01_baselines_core7 | 30 | 14 | 0.388 | -0.403 | -5.959 | - | 11 | FAIL | logreg+rf {'model': 'logreg', 'C': 1.0, 'max_iter': 2000, 'class_weight': 'balanced'} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 5 | 1 | 0.000 | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 15 | 3 | 0.000 | -7.450 | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B01_baselines_core7 | 5 | 13 | 0.321 | -0.491 | -7.346 | - | 20 | FAIL | logreg+rf {'model': 'rf', 'n_estimators': 300, 'max_depth': 6, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 15 | 4 | 0.000 | -182.371 | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 15 | 1 | 0.000 | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B01_baselines_core7 | 15 | 5 | 2.501 | 0.323 | - | - | - | FAIL | logreg+rf {'model': 'rf', 'n_estimators': 300, 'max_depth': 6, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 30 | 1 | 0.000 | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B01_baselines_core7 | 30 | 5 | 5.250 | 0.491 | - | - | - | FAIL | logreg+rf {'model': 'rf', 'n_estimators': 300, 'max_depth': 6, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 5 | 1 | 0.000 | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 15 | 1 | 0.000 | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 5 | 2 | 0.000 | -1.684 | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 15 | 2 | 0.000 | -1.027 | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B02_xgb_core7 | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B03_xgb_cross_asset | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B05_temporal_core7 | 5 | 0 | - | - | - | - | - | FAIL | torch-temporal {'model': 'temporal-cnn', 'seq_len': 60, 'hidden': 64} thr=0.6 |
| 2026-07-17 | B05_temporal_core7 | 15 | 0 | - | - | - | - | - | FAIL | torch-temporal {'model': 'temporal-cnn', 'seq_len': 60, 'hidden': 64} thr=0.6 |
| 2026-07-17 | B05_temporal_core7 | 30 | 0 | - | - | - | - | - | FAIL | torch-temporal {'model': 'temporal-cnn', 'seq_len': 60, 'hidden': 64} thr=0.6 |
| 2026-07-17 | B05_temporal_core7 | 5 | 0 | - | - | - | - | - | FAIL | torch-temporal {'model': 'temporal-gru', 'seq_len': 60, 'hidden': 64} thr=0.6 |
| 2026-07-17 | B05_temporal_core7 | 15 | 0 | - | - | - | - | - | FAIL | torch-temporal {'model': 'temporal-gru', 'seq_len': 60, 'hidden': 64} thr=0.6 |
| 2026-07-17 | B05_temporal_core7 | 30 | 0 | - | - | - | - | - | FAIL | torch-temporal {'model': 'temporal-gru', 'seq_len': 60, 'hidden': 64} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 5} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 5 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 15 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-17 | B04_xgb_sentiment | 30 | 0 | - | - | - | - | - | FAIL | xgboost-cuda {'model': 'xgb', 'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 50} thr=0.6 |
| 2026-07-18 | B06_direction_rank | 5 | 0 | - | - | - | - | - | FAIL | logreg+rf+xgb direction/rank IC=None t=None |
| 2026-07-18 | B06_direction_rank | 15 | 0 | - | - | - | - | - | FAIL | logreg+rf+xgb direction/rank IC=None t=None |
| 2026-07-18 | B06_direction_rank | 30 | 0 | - | - | - | - | - | FAIL | logreg+rf+xgb direction/rank IC=None t=None |
| 2026-07-18 | B06_direction_rank | 5 | 0 | - | - | - | - | - | FAIL | logreg+rf+xgb direction/rank IC=None t=None |
| 2026-07-18 | B06_direction_rank | 15 | 0 | - | - | - | - | - | FAIL | logreg+rf+xgb direction/rank IC=None t=None |
| 2026-07-18 | B06_direction_rank | 30 | 0 | - | - | - | - | - | FAIL | logreg+rf+xgb direction/rank IC=None t=None |
| 2026-07-18 | B06_direction_rank | 5 | 0 | - | - | - | - | - | FAIL | logreg+rf+xgb direction/rank IC=None t=None |
| 2026-07-18 | B06_direction_rank | 15 | 0 | - | - | - | - | - | FAIL | logreg+rf+xgb direction/rank IC=None t=None |
| 2026-07-18 | B06_direction_rank | 30 | 0 | - | - | - | - | - | FAIL | logreg+rf+xgb direction/rank IC=None t=None |
| 2026-07-18 | B06_direction_rank | 5 | 13038 | 0.210 | -0.636 | -249.765 | 0.000 | 126 | FAIL | logreg+rf+xgb direction/rank IC=0.01045 t=1.227 |
| 2026-07-18 | B06_direction_rank | 15 | 4254 | 0.249 | -0.564 | -136.127 | 0.000 | 127 | FAIL | logreg+rf+xgb direction/rank IC=-0.0033 t=-0.419 |
| 2026-07-18 | B06_direction_rank | 30 | 2069 | 0.279 | -0.484 | -79.033 | 0.000 | 128 | FAIL | logreg+rf+xgb direction/rank IC=0.01098 t=1.166 |
| 2026-07-18 | B06_direction_rank | 5 | 15598 | 0.254 | -0.552 | -232.756 | 0.000 | 129 | FAIL | logreg+rf+xgb direction/rank IC=0.10576 t=9.275 |
| 2026-07-18 | B06_direction_rank | 15 | 5060 | 0.170 | -0.690 | -71.987 | 0.000 | 130 | FAIL | logreg+rf+xgb direction/rank IC=0.0266 t=2.235 |
| 2026-07-18 | B06_direction_rank | 30 | 2362 | 0.163 | -0.683 | -45.760 | 0.000 | 131 | FAIL | logreg+rf+xgb direction/rank IC=0.01947 t=1.505 |
| 2026-07-18 | B06_direction_rank | 5 | 16212 | 0.243 | -0.562 | -292.530 | 0.000 | 132 | FAIL | logreg+rf+xgb direction/rank IC=0.08502 t=8.829 |
| 2026-07-18 | B06_direction_rank | 15 | 5182 | 0.330 | -0.419 | -120.312 | 0.000 | 133 | FAIL | logreg+rf+xgb direction/rank IC=0.03693 t=3.819 |
                                                                                                                                                  | 2026-07-18 | B06_direction_rank | 5 | 13038 | 0.210 | -0.636 | -48.939 | 0.000 | 1 | FAIL | logreg+xgb direction/rank IC=0.01045 t=1.227 |
| 2026-07-18 | B06_direction_rank | 15 | 4254 | 0.249 | -0.564 | -46.421 | 0.000 | 2 | FAIL | logreg+xgb direction/rank IC=-0.0033 t=-0.419 |
| 2026-07-18 | B06_direction_rank | 30 | 2069 | 0.279 | -0.484 | -34.071 | 0.000 | 3 | FAIL | logreg+xgb direction/rank IC=0.01098 t=1.166 |
| 2026-07-18 | B06_direction_rank | 5 | 16212 | 0.243 | -0.562 | -148.056 | 0.000 | 4 | FAIL | logreg+xgb direction/rank IC=0.08502 t=8.829 |
| 2026-07-18 | B06_direction_rank | 15 | 5182 | 0.330 | -0.419 | -63.603 | 0.000 | 5 | FAIL | logreg+xgb direction/rank IC=0.03693 t=3.819 |
| 2026-07-18 | B06_direction_rank | 30 | 2775 | 0.314 | -0.414 | -51.428 | 0.000 | 6 | FAIL | logreg+xgb direction/rank IC=0.02456 t=2.51 |
| 2026-07-18 | B06_direction_rank_4Hour | 5 | 83 | 1.012 | 0.005 | 0.049 | 0.484 | 1 | PASS | logreg+xgb direction/rank IC=-0.03514 t=-0.871 |
| 2026-07-18 | B06_direction_rank_4Hour | 15 | 36 | 1.535 | 0.177 | 1.084 | - | 1 | FAIL | logreg+xgb direction/rank IC=-0.03275 t=-0.521 |
| 2026-07-18 | B06_direction_rank_4Hour | 30 | 27 | 2.416 | 0.352 | -0.885 | - | 2 | FAIL | logreg+xgb direction/rank IC=-0.06794 t=-0.667 |
| 2026-07-18 | B06_direction_rank_4Hour | 5 | 115 | 0.819 | -0.086 | -9.887 | 0.179 | 3 | FAIL | logreg+xgb direction/rank IC=-0.04748 t=-1.29 |
| 2026-07-18 | B06_direction_rank_4Hour | 15 | 50 | 1.002 | 0.001 | -7.359 | 0.500 | 4 | FAIL | logreg+xgb direction/rank IC=-0.07292 t=-1.031 |
| 2026-07-18 | B06_direction_rank_4Hour | 30 | 37 | 0.881 | -0.056 | -7.346 | - | 5 | FAIL | logreg+xgb direction/rank IC=-0.16546 t=-1.382 |
| 2026-07-18 | B07_magnitude_4Hour | 5 | 76 | 0.839 | -0.075 | -11.618 | 0.230 | 6 | FAIL | xgboost-regressor magnitude/rank IC=0.05919 t=1.44 |
| 2026-07-18 | B07_magnitude_4Hour | 15 | 41 | 1.541 | 0.177 | -7.782 | 0.230 | 7 | FAIL | xgboost-regressor magnitude/rank IC=0.05915 t=0.914 |
| 2026-07-18 | B07_magnitude_4Hour | 30 | 29 | 1.095 | 0.041 | -7.556 | - | 8 | FAIL | xgboost-regressor magnitude/rank IC=0.02069 t=0.237 |
| 2026-07-18 | B07_magnitude_4Hour | 5 | 78 | 0.854 | -0.068 | -13.704 | 0.218 | 9 | FAIL | xgboost-regressor magnitude/rank IC=0.10403 t=2.696 |
| 2026-07-18 | B07_magnitude_4Hour | 15 | 34 | 0.925 | -0.033 | -9.147 | - | 10 | FAIL | xgboost-regressor magnitude/rank IC=0.0532 t=1.051 |
| 2026-07-18 | B07_magnitude_4Hour | 30 | 33 | 1.062 | 0.026 | -9.069 | - | 11 | FAIL | xgboost-regressor magnitude/rank IC=0.08176 t=0.926 |
| 2026-07-18 | B08_xsect_4Hour | 5 | 341 | 0.864 | -0.034 | -35.037 | 0.222 | 12 | FAIL | xgboost-xsect xsect/absolute IC=0.04872 t=1.763 |
| 2026-07-18 | B08_xsect_4Hour | 15 | 124 | 1.676 | 0.137 | -22.121 | 0.068 | 13 | FAIL | xgboost-xsect xsect/absolute IC=0.1347 t=2.76 |
| 2026-07-18 | B08_xsect_4Hour | 30 | 70 | 1.300 | 0.081 | -13.672 | 0.155 | 14 | FAIL | xgboost-xsect xsect/absolute IC=0.12593 t=1.8 |
| 2026-07-18 | B06_direction_rank_1Hour | 5 | 376 | 1.049 | 0.019 | 0.373 | 0.290 | 1 | PASS | logreg+xgb direction/rank IC=-0.00305 t=-0.195 |
| 2026-07-18 | B06_direction_rank_1Hour | 15 | 138 | 0.963 | -0.015 | -6.219 | 0.488 | 2 | FAIL | logreg+xgb direction/rank IC=-0.00725 t=-0.205 |
| 2026-07-18 | B06_direction_rank_1Hour | 30 | 71 | 0.910 | -0.039 | -7.419 | 0.325 | 3 | FAIL | logreg+xgb direction/rank IC=0.02271 t=0.534 |
| 2026-07-18 | B06_direction_rank_1Hour | 5 | 334 | 1.005 | 0.002 | -19.173 | 0.456 | 4 | FAIL | logreg+xgb direction/rank IC=-0.02376 t=-1.211 |
| 2026-07-18 | B06_direction_rank_1Hour | 15 | 132 | 0.713 | -0.109 | -16.715 | 0.115 | 5 | FAIL | logreg+xgb direction/rank IC=0.00252 t=0.09 |
| 2026-07-18 | B06_direction_rank_1Hour | 30 | 66 | 1.473 | 0.159 | -9.451 | 0.024 | 6 | FAIL | logreg+xgb direction/rank IC=0.03478 t=1.035 |
| 2026-07-18 | B07_magnitude_1Hour | 5 | 308 | 0.688 | -0.141 | -25.830 | 0.000 | 7 | FAIL | xgboost-regressor magnitude/rank IC=0.09707 t=4.794 |
| 2026-07-18 | B07_magnitude_1Hour | 15 | 110 | 0.929 | -0.029 | -15.457 | 0.409 | 8 | FAIL | xgboost-regressor magnitude/rank IC=0.0562 t=1.85 |
| 2026-07-18 | B07_magnitude_1Hour | 30 | 63 | 0.713 | -0.121 | -12.318 | 0.083 | 9 | FAIL | xgboost-regressor magnitude/rank IC=-0.01512 t=-0.445 |
| 2026-07-18 | B07_magnitude_1Hour | 5 | 277 | 0.945 | -0.022 | -26.387 | 0.441 | 10 | FAIL | xgboost-regressor magnitude/rank IC=0.10232 t=4.807 |
| 2026-07-18 | B07_magnitude_1Hour | 15 | 117 | 1.011 | 0.004 | -17.438 | 0.460 | 11 | FAIL | xgboost-regressor magnitude/rank IC=0.0364 t=1.136 |
| 2026-07-18 | B07_magnitude_1Hour | 30 | 67 | 0.952 | -0.018 | -13.638 | 0.472 | 12 | FAIL | xgboost-regressor magnitude/rank IC=0.00254 t=0.062 |
| 2026-07-18 | B08_xsect_1Hour | 5 | 1510 | 0.956 | -0.010 | -70.865 | 0.345 | 13 | FAIL | xgboost-xsect xsect/absolute IC=0.06126 t=4.801 |
| 2026-07-18 | B08_xsect_1Hour | 15 | 510 | 1.085 | 0.018 | -36.254 | 0.333 | 14 | FAIL | xgboost-xsect xsect/absolute IC=0.10699 t=6.005 |
| 2026-07-18 | B08_xsect_1Hour | 30 | 259 | 1.111 | 0.031 | -26.058 | 0.234 | 15 | FAIL | xgboost-xsect xsect/absolute IC=0.11987 t=4.979 |
| 2026-07-18 | B06_direction_rank_15Min | 5 | 774 | 0.692 | -0.133 | -4.035 | 0.000 | 1 | FAIL | logreg+xgb direction/rank IC=-0.00306 t=-0.198 |
| 2026-07-18 | B06_direction_rank_15Min | 15 | 145 | 0.667 | -0.164 | -7.729 | 0.036 | 2 | FAIL | logreg+xgb direction/rank IC=-0.01131 t=-0.334 |
| 2026-07-18 | B06_direction_rank_15Min | 30 | 0 | - | - | - | - | - | FAIL | logreg+xgb direction/rank IC=None t=None |
| 2026-07-18 | B06_direction_rank_15Min | 5 | 859 | 0.572 | -0.216 | -34.002 | 0.000 | 4 | FAIL | logreg+xgb direction/rank IC=-0.02783 t=-2.662 |
| 2026-07-18 | B06_direction_rank_15Min | 15 | 155 | 0.604 | -0.212 | -16.659 | 0.000 | 5 | FAIL | logreg+xgb direction/rank IC=0.01073 t=0.452 |
| 2026-07-18 | B06_direction_rank_15Min | 30 | 0 | - | - | - | - | - | FAIL | logreg+xgb direction/rank IC=None t=None |
| 2026-07-18 | B07_magnitude_15Min | 5 | 1057 | 0.599 | nan | nan | 0.000 | 7 | FAIL | xgboost-regressor magnitude/rank IC=0.10031 t=7.924 |
| 2026-07-18 | B07_magnitude_15Min | 15 | 424 | 0.463 | nan | nan | 0.000 | 8 | FAIL | xgboost-regressor magnitude/rank IC=0.05956 t=2.588 |
| 2026-07-18 | B07_magnitude_15Min | 30 | 0 | - | - | - | - | - | FAIL | xgboost-regressor magnitude/rank IC=None t=None |
| 2026-07-18 | B07_magnitude_15Min | 5 | 1035 | 0.653 | nan | nan | 0.000 | 10 | FAIL | xgboost-regressor magnitude/rank IC=0.10361 t=7.781 |
| 2026-07-18 | B07_magnitude_15Min | 15 | 479 | 0.407 | nan | nan | 0.000 | 11 | FAIL | xgboost-regressor magnitude/rank IC=0.0516 t=1.642 |
| 2026-07-18 | B07_magnitude_15Min | 30 | 0 | - | - | - | - | - | FAIL | xgboost-regressor magnitude/rank IC=None t=None |
| 2026-07-18 | B08_xsect_15Min | 5 | 3670 | 0.894 | -0.036 | -104.861 | 0.012 | 13 | FAIL | xgboost-xsect xsect/absolute IC=-0.01454 t=-2.498 |
| 2026-07-18 | B08_xsect_15Min | 15 | 642 | 0.889 | -0.038 | -44.113 | 0.139 | 14 | FAIL | xgboost-xsect xsect/absolute IC=-0.01553 t=-0.77 |
| 2026-07-18 | B08_xsect_15Min | 30 | 0 | - | - | - | - | - | FAIL | xgboost-xsect xsect/absolute IC=None t=None |
| 2026-07-18 | B06_direction_rank_5Min | 5 | 2704 | 0.453 | -0.313 | -13.425 | 0.000 | 1 | FAIL | logreg+xgb direction/rank IC=0.01921 t=2.449 |
| 2026-07-18 | B06_direction_rank_5Min | 15 | 799 | 0.568 | -0.188 | -22.009 | 0.000 | 2 | FAIL | logreg+xgb direction/rank IC=0.02555 t=1.422 |
| 2026-07-18 | B06_direction_rank_5Min | 30 | 309 | 0.597 | -0.211 | -18.101 | 0.000 | 3 | FAIL | logreg+xgb direction/rank IC=0.02245 t=0.694 |
| 2026-07-18 | B06_direction_rank_5Min | 5 | 2414 | 0.397 | -0.355 | -51.589 | 0.000 | 4 | FAIL | logreg+xgb direction/rank IC=0.02047 t=5.339 |
| 2026-07-18 | B06_direction_rank_5Min | 15 | 989 | 0.539 | -0.230 | -34.188 | 0.000 | 5 | FAIL | logreg+xgb direction/rank IC=0.00626 t=0.466 |
| 2026-07-18 | B06_direction_rank_5Min | 30 | 355 | 0.643 | -0.182 | -25.728 | 0.000 | 6 | FAIL | logreg+xgb direction/rank IC=0.0104 t=0.711 |
| 2026-07-18 | B07_magnitude_5Min | 5 | 2411 | 0.327 | nan | nan | 0.000 | 7 | FAIL | xgboost-regressor magnitude/rank IC=0.17931 t=32.425 |
| 2026-07-18 | B07_magnitude_5Min | 15 | 1033 | 0.252 | nan | nan | 0.000 | 8 | FAIL | xgboost-regressor magnitude/rank IC=0.19789 t=16.408 |
| 2026-07-18 | B07_magnitude_5Min | 30 | 588 | 0.081 | nan | nan | 0.000 | 9 | FAIL | xgboost-regressor magnitude/rank IC=0.19364 t=11.235 |
| 2026-07-18 | B07_magnitude_5Min | 5 | 2302 | 0.363 | nan | nan | 0.000 | 10 | FAIL | xgboost-regressor magnitude/rank IC=0.18174 t=27.159 |
| 2026-07-18 | B07_magnitude_5Min | 15 | 1083 | 0.194 | nan | nan | 0.000 | 11 | FAIL | xgboost-regressor magnitude/rank IC=0.201 t=22.206 |
| 2026-07-18 | B07_magnitude_5Min | 30 | 633 | 0.045 | nan | nan | 0.000 | 12 | FAIL | xgboost-regressor magnitude/rank IC=0.20369 t=12.29 |
| 2026-07-18 | B08_xsect_5Min | 5 | 12748 | 0.836 | -0.054 | -195.192 | 0.000 | 13 | FAIL | xgboost-xsect xsect/absolute IC=-0.00499 t=-0.993 |
| 2026-07-18 | B08_xsect_5Min | 15 | 3670 | 0.867 | -0.041 | -105.985 | 0.000 | 14 | FAIL | xgboost-xsect xsect/absolute IC=-0.0113 t=-1.052 |
| 2026-07-18 | B08_xsect_5Min | 30 | 1400 | 0.851 | -0.046 | -65.018 | 0.040 | 15 | FAIL | xgboost-xsect xsect/absolute IC=-0.00936 t=-0.535 |
| 2026-07-19 | B09_kronos_dir_4Hour | 5 | 0 | - | - | - | - | - | FAIL | kronos-direction xsect/rank IC=None t=None |
| 2026-07-19 | B09_kronos_dir_4Hour | 15 | 0 | - | - | - | - | - | FAIL | kronos-direction xsect/rank IC=None t=None |
| 2026-07-19 | B09_kronos_dir_4Hour | 30 | 0 | - | - | - | - | - | FAIL | kronos-direction xsect/rank IC=None t=None |
| 2026-07-19 | B10_kronos_mag_4Hour | 5 | 19 | 1.457 | 0.168 | -3.736 | - | 4 | FAIL | kronos-magnitude magnitude/rank IC=None t=None |
| 2026-07-19 | B10_kronos_mag_4Hour | 15 | 19 | 1.177 | 0.073 | -4.804 | - | 5 | FAIL | kronos-magnitude magnitude/rank IC=None t=None |
| 2026-07-19 | B10_kronos_mag_4Hour | 30 | 18 | 0.901 | -0.045 | -5.471 | - | 6 | FAIL | kronos-magnitude magnitude/rank IC=None t=None |
| 2026-07-19 | B09_kronos_dir_1Hour | 5 | 1 | inf | - | - | - | - | FAIL | kronos-direction xsect/rank IC=0.03869 t=0.66 |
| 2026-07-19 | B09_kronos_dir_1Hour | 15 | 1 | inf | - | - | - | - | FAIL | kronos-direction xsect/rank IC=None t=None |
| 2026-07-19 | B09_kronos_dir_1Hour | 30 | 1 | inf | - | - | - | - | FAIL | kronos-direction xsect/rank IC=None t=None |
| 2026-07-19 | B10_kronos_mag_1Hour | 5 | 20 | 1.905 | 0.274 | -3.472 | - | 4 | FAIL | kronos-magnitude magnitude/rank IC=None t=None |
| 2026-07-19 | B10_kronos_mag_1Hour | 15 | 20 | 1.187 | 0.076 | -4.918 | - | 5 | FAIL | kronos-magnitude magnitude/rank IC=None t=None |
| 2026-07-19 | B10_kronos_mag_1Hour | 30 | 20 | 1.867 | 0.272 | -4.506 | - | 6 | FAIL | kronos-magnitude magnitude/rank IC=None t=None |
| 2026-07-21 | B09_kronos_dir_4Hour | 5 | 2 | 0.000 | -0.871 | - | - | - | FAIL | kronos-direction xsect/rank IC=0.07328 t=1.146 |
| 2026-07-21 | B09_kronos_dir_4Hour | 15 | 2 | 0.000 | -1.415 | - | - | - | FAIL | kronos-direction xsect/rank IC=0.02989 t=0.371 |
| 2026-07-21 | B09_kronos_dir_4Hour | 30 | 2 | 2.390 | 0.290 | - | - | - | FAIL | kronos-direction xsect/rank IC=0.13711 t=1.805 |
| 2026-07-21 | B10_kronos_mag_4Hour | 5 | 33 | 0.865 | -0.066 | -9.171 | - | 10 | FAIL | kronos-magnitude magnitude/rank IC=0.06042 t=0.624 |
| 2026-07-21 | B10_kronos_mag_4Hour | 15 | 32 | 1.365 | 0.103 | -9.772 | - | 11 | FAIL | kronos-magnitude magnitude/rank IC=-0.04204 t=-0.452 |
| 2026-07-21 | B10_kronos_mag_4Hour | 30 | 29 | 1.806 | 0.149 | -10.907 | - | 12 | FAIL | kronos-magnitude magnitude/rank IC=0.1006 t=1.074 |
| 2026-07-22 | B09_kronos_dir_4Hour | 5 | 2 | 0.000 | -0.871 | - | - | - | FAIL | kronos-direction xsect/rank IC=0.07328 t=1.146 |
| 2026-07-22 | B09_kronos_dir_4Hour | 15 | 2 | 0.000 | -1.415 | - | - | - | FAIL | kronos-direction xsect/rank IC=0.02989 t=0.371 |
| 2026-07-22 | B09_kronos_dir_4Hour | 30 | 2 | 2.390 | 0.290 | - | - | - | FAIL | kronos-direction xsect/rank IC=0.13711 t=1.805 |
| 2026-07-22 | B10_kronos_mag_4Hour | 5 | 33 | 0.865 | -0.066 | -10.434 | - | 16 | FAIL | kronos-magnitude magnitude/rank IC=0.06042 t=0.624 |
| 2026-07-22 | B10_kronos_mag_4Hour | 15 | 32 | 1.365 | 0.103 | -11.096 | - | 17 | FAIL | kronos-magnitude magnitude/rank IC=-0.04204 t=-0.452 |
| 2026-07-22 | B10_kronos_mag_4Hour | 30 | 29 | 1.806 | 0.149 | -12.267 | - | 18 | FAIL | kronos-magnitude magnitude/rank IC=0.1006 t=1.074 |
| 2026-07-22 | B09_kronos_dir_1Hour | 5 | 11 | 0.353 | -0.344 | -5.592 | - | 7 | FAIL | kronos-direction xsect/rank IC=-0.01175 t=-0.439 |
| 2026-07-22 | B09_kronos_dir_1Hour | 15 | 11 | 2.029 | 0.229 | -4.272 | - | 8 | FAIL | kronos-direction xsect/rank IC=0.04719 t=1.646 |
| 2026-07-22 | B09_kronos_dir_1Hour | 30 | 11 | 0.550 | -0.203 | -6.179 | - | 9 | FAIL | kronos-direction xsect/rank IC=0.04205 t=1.381 |
| 2026-07-22 | B10_kronos_mag_1Hour | 5 | 105 | 0.951 | -0.023 | -16.270 | 0.258 | 10 | FAIL | kronos-magnitude magnitude/rank IC=0.04065 t=1.049 |
| 2026-07-22 | B10_kronos_mag_1Hour | 15 | 108 | 0.885 | -0.050 | -17.909 | 0.016 | 11 | FAIL | kronos-magnitude magnitude/rank IC=-0.02791 t=-0.632 |
| 2026-07-22 | B10_kronos_mag_1Hour | 30 | 108 | 0.739 | -0.119 | -18.790 | 0.079 | 12 | FAIL | kronos-magnitude magnitude/rank IC=-0.01444 t=-0.417 |
| 2026-07-22 | B11_kronos_finetuned_1Hour | 5 | 39 | 1.179 | 0.070 | 0.435 | - | 1 | FAIL | kronos-finetuned magnitude/rank IC=-0.06422 t=-1.168 |
| 2026-07-22 | B11_kronos_finetuned_1Hour | 15 | 39 | 1.374 | 0.137 | -2.365 | - | 2 | FAIL | kronos-finetuned magnitude/rank IC=0.00027 t=0.004 |
| 2026-07-22 | B11_kronos_finetuned_1Hour | 30 | 36 | 0.727 | -0.133 | -5.708 | - | 3 | FAIL | kronos-finetuned magnitude/rank IC=-0.07308 t=-1.018 |
| 2026-07-26 | B10_kronos_mag_15Min | 5 | 125 | 0.954 | nan | nan | 0.397 | 1 | FAIL | kronos-magnitude magnitude/rank IC=0.0634 t=1.705 |
| 2026-07-26 | B10_kronos_mag_15Min | 15 | 125 | 1.401 | nan | nan | 0.163 | 2 | FAIL | kronos-magnitude magnitude/rank IC=0.02842 t=0.61 |
| 2026-07-26 | B10_kronos_mag_15Min | 30 | 125 | 0.000 | nan | nan | 0.000 | 3 | FAIL | kronos-magnitude magnitude/rank IC=None t=None |
| 2026-07-26 | B09_kronos_dir_15Min | 5 | 188 | 1.313 | 0.096 | -13.573 | 0.000 | 4 | FAIL | kronos-direction direction/rank IC=-0.01299 t=-0.433 |
| 2026-07-26 | B09_kronos_dir_15Min | 15 | 102 | 1.439 | 0.155 | -10.363 | 0.048 | 5 | FAIL | kronos-direction direction/rank IC=0.03313 t=0.914 |
| 2026-07-26 | B09_kronos_dir_15Min | 30 | 0 | - | - | - | - | - | FAIL | kronos-direction direction/rank IC=None t=None |
| 2026-07-26 | B10_kronos_mag_5Min | 5 | 177 | 0.708 | nan | nan | 0.083 | 1 | FAIL | kronos-magnitude magnitude/rank IC=0.01046 t=0.368 |
| 2026-07-26 | B10_kronos_mag_5Min | 15 | 176 | 0.834 | nan | nan | 0.119 | 2 | FAIL | kronos-magnitude magnitude/rank IC=-0.02553 t=-1.237 |
| 2026-07-26 | B10_kronos_mag_5Min | 30 | 178 | 1.012 | nan | nan | 0.429 | 3 | FAIL | kronos-magnitude magnitude/rank IC=-0.12775 t=-4.225 |
| 2026-07-26 | B09_kronos_dir_5Min | 5 | 269 | 0.968 | -0.012 | -17.299 | 0.417 | 4 | FAIL | kronos-direction direction/rank IC=0.00579 t=0.263 |
| 2026-07-26 | B09_kronos_dir_5Min | 15 | 228 | 0.955 | -0.017 | -18.004 | 0.492 | 5 | FAIL | kronos-direction direction/rank IC=0.02543 t=1.093 |
| 2026-07-26 | B09_kronos_dir_5Min | 30 | 168 | 0.915 | -0.034 | -16.717 | 0.441 | 6 | FAIL | kronos-direction direction/rank IC=0.02204 t=0.723 |
| 2026-07-26 | B10_kronos_mag_1Hour_btc | 5 | 36 | 0.220 | -0.466 | -3.885 | - | 1 | FAIL | kronos-magnitude magnitude/rank IC=-0.16317 t=-2.293 |
| 2026-07-26 | B10_kronos_mag_1Hour_btc | 15 | 36 | 0.721 | -0.148 | -3.864 | - | 2 | FAIL | kronos-magnitude magnitude/rank IC=-0.08411 t=-1.437 |
| 2026-07-26 | B10_kronos_mag_1Hour_btc | 30 | 34 | 0.807 | -0.088 | -5.086 | - | 3 | FAIL | kronos-magnitude magnitude/rank IC=-0.04502 t=-0.558 |
| 2026-07-26 | B09_kronos_dir_1Hour_btc | 5 | 36 | 0.687 | -0.173 | -6.992 | - | 4 | FAIL | kronos-direction direction/rank IC=0.02589 t=0.394 |
| 2026-07-26 | B09_kronos_dir_1Hour_btc | 15 | 36 | 0.480 | -0.327 | -8.721 | - | 5 | FAIL | kronos-direction direction/rank IC=-0.07232 t=-1.172 |
| 2026-07-26 | B09_kronos_dir_1Hour_btc | 30 | 34 | 0.445 | -0.333 | -9.897 | - | 6 | FAIL | kronos-direction direction/rank IC=-0.08071 t=-1.118 |
| 2026-07-26 | B10_kronos_mag_1Min_btc | 5 | 63 | 1.029 | 0.011 | 0.085 | 0.441 | 1 | PASS | kronos-magnitude magnitude/rank IC=0.00475 t=0.096 |
| 2026-07-26 | B10_kronos_mag_1Min_btc | 15 | 63 | 0.771 | -0.095 | -4.681 | 0.127 | 2 | FAIL | kronos-magnitude magnitude/rank IC=-0.02419 t=-0.542 |
| 2026-07-26 | B10_kronos_mag_1Min_btc | 30 | 63 | 0.509 | nan | nan | 0.131 | 3 | FAIL | kronos-magnitude magnitude/rank IC=-0.01522 t=-0.3 |
| 2026-07-26 | B09_kronos_dir_1Min_btc | 5 | 63 | 0.446 | -0.345 | -10.610 | 0.000 | 4 | FAIL | kronos-direction direction/rank IC=-0.07131 t=-1.454 |
| 2026-07-26 | B09_kronos_dir_1Min_btc | 15 | 59 | 0.557 | -0.262 | -10.684 | 0.032 | 5 | FAIL | kronos-direction direction/rank IC=-0.05681 t=-0.954 |
| 2026-07-26 | B09_kronos_dir_1Min_btc | 30 | 53 | 0.553 | -0.154 | -12.493 | 0.266 | 6 | FAIL | kronos-direction direction/rank IC=-0.04201 t=-0.578 |
| 2026-07-26 | B09_kronos_dir_1Hour_all23 | 5 | 207 | 0.818 | -0.064 | -0.941 | 0.182 | 1 | FAIL | kronos-direction xsect/rank IC=0.00146 t=0.091 |
| 2026-07-26 | B09_kronos_dir_1Hour_all23 | 15 | 207 | 1.059 | 0.018 | -7.193 | 0.444 | 2 | FAIL | kronos-direction xsect/rank IC=0.03201 t=1.732 |
| 2026-07-26 | B09_kronos_dir_1Hour_all23 | 30 | 207 | 1.059 | 0.018 | -12.059 | 0.421 | 3 | FAIL | kronos-direction xsect/rank IC=0.0562 t=3.07 |
| 2026-07-26 | B10_kronos_mag_1Hour_all23 | 5 | 477 | 0.882 | -0.048 | -24.302 | 0.139 | 4 | FAIL | kronos-magnitude magnitude/rank IC=-0.02005 t=-1.058 |
| 2026-07-26 | B10_kronos_mag_1Hour_all23 | 15 | 477 | 0.910 | -0.033 | -28.182 | 0.167 | 5 | FAIL | kronos-magnitude magnitude/rank IC=-0.05135 t=-2.029 |
| 2026-07-26 | B10_kronos_mag_1Hour_all23 | 30 | 470 | 0.940 | -0.024 | -28.631 | 0.270 | 6 | FAIL | kronos-magnitude magnitude/rank IC=-0.05269 t=-2.038 |
| 2026-07-26 | B10_kronos_mag_1Hour_btc | 5 | 144 | 0.741 | -0.121 | -17.627 | 0.000 | 7 | FAIL | kronos-magnitude magnitude/rank IC=-0.08862 t=-3.487 |
| 2026-07-26 | B10_kronos_mag_1Hour_btc | 15 | 144 | 0.877 | -0.055 | -18.039 | 0.314 | 8 | FAIL | kronos-magnitude magnitude/rank IC=-0.05047 t=-1.797 |
| 2026-07-26 | B10_kronos_mag_1Hour_btc | 30 | 144 | 0.922 | -0.032 | -18.473 | 0.309 | 9 | FAIL | kronos-magnitude magnitude/rank IC=-0.06033 t=-3.038 |
| 2026-07-26 | B09_kronos_dir_1Hour_btc | 5 | 144 | 1.015 | 0.006 | -18.774 | 0.448 | 10 | FAIL | kronos-direction direction/rank IC=0.03302 t=1.209 |
| 2026-07-26 | B09_kronos_dir_1Hour_btc | 15 | 144 | 0.985 | -0.006 | -19.483 | 0.389 | 11 | FAIL | kronos-direction direction/rank IC=0.04684 t=1.685 |
| 2026-07-26 | B09_kronos_dir_1Hour_btc | 30 | 144 | 0.982 | -0.007 | -19.975 | 0.337 | 12 | FAIL | kronos-direction direction/rank IC=0.01174 t=0.473 |
| 2026-07-26 | B10_kronos_mag_1Min_btc | 5 | 259 | 0.999 | -0.000 | -22.271 | 0.500 | 7 | FAIL | kronos-magnitude magnitude/rank IC=-0.00702 t=-0.239 |
| 2026-07-26 | B10_kronos_mag_1Min_btc | 15 | 259 | 0.999 | -0.000 | -23.433 | 0.484 | 8 | FAIL | kronos-magnitude magnitude/rank IC=-0.01725 t=-0.7 |
| 2026-07-26 | B10_kronos_mag_1Min_btc | 30 | 259 | 0.828 | -0.058 | -25.613 | 0.429 | 9 | FAIL | kronos-magnitude magnitude/rank IC=-0.05903 t=-3.012 |
| 2026-07-26 | B09_kronos_dir_1Min_btc | 5 | 259 | 0.646 | -0.168 | -24.745 | 0.000 | 10 | FAIL | kronos-direction direction/rank IC=-0.01577 t=-0.708 |
| 2026-07-26 | B09_kronos_dir_1Min_btc | 15 | 259 | 0.535 | -0.245 | -31.577 | 0.000 | 11 | FAIL | kronos-direction direction/rank IC=-0.03897 t=-1.552 |
| 2026-07-26 | B09_kronos_dir_1Min_btc | 30 | 259 | 0.551 | -0.164 | -36.940 | 0.000 | 12 | FAIL | kronos-direction direction/rank IC=-0.01074 t=-0.417 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16683 | 0.275 | -0.498 | -32.215 | 0.000 | 8 | FAIL | xgboost-cuda direction/rank IC=0.09411 t=7.543 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5288 | 0.268 | -0.509 | -19.542 | 0.000 | 9 | FAIL | xgboost-cuda direction/rank IC=0.0362 t=4.015 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2720 | 0.299 | -0.440 | -12.366 | 0.000 | 10 | FAIL | xgboost-cuda direction/rank IC=0.02632 t=2.261 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16604 | 0.267 | -0.522 | -35.941 | 0.000 | 11 | FAIL | xgboost-cuda direction/rank IC=0.09177 t=8.571 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5340 | 0.265 | -0.517 | -20.054 | 0.000 | 12 | FAIL | xgboost-cuda direction/rank IC=0.03538 t=3.643 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2670 | 0.280 | -0.465 | -13.050 | 0.000 | 13 | FAIL | xgboost-cuda direction/rank IC=0.01762 t=1.521 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 15489 | 0.243 | -0.565 | -51.309 | 0.000 | 14 | FAIL | xgboost-cuda direction/rank IC=0.08389 t=6.45 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5106 | 0.276 | -0.506 | -19.392 | 0.000 | 15 | FAIL | xgboost-cuda direction/rank IC=0.03429 t=3.149 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2872 | 0.276 | -0.483 | -19.402 | 0.000 | 16 | FAIL | xgboost-cuda direction/rank IC=0.02197 t=1.979 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 15915 | 0.260 | -0.518 | -32.020 | 0.000 | 17 | FAIL | xgboost-cuda direction/rank IC=0.08617 t=7.263 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5346 | 0.288 | -0.490 | -19.287 | 0.000 | 18 | FAIL | xgboost-cuda direction/rank IC=0.0335 t=3.297 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2898 | 0.294 | -0.463 | -16.695 | 0.000 | 19 | FAIL | xgboost-cuda direction/rank IC=0.01577 t=1.354 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16679 | 0.261 | -0.528 | -41.462 | 0.000 | 20 | FAIL | xgboost-cuda direction/rank IC=0.08014 t=6.27 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5260 | 0.310 | -0.451 | -19.317 | 0.000 | 21 | FAIL | xgboost-cuda direction/rank IC=0.03053 t=3.046 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2622 | 0.315 | -0.428 | -13.836 | 0.000 | 22 | FAIL | xgboost-cuda direction/rank IC=0.0241 t=2.507 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16212 | 0.243 | -0.562 | -53.486 | 0.000 | 23 | FAIL | xgboost-cuda direction/rank IC=0.08502 t=8.829 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5182 | 0.330 | -0.419 | -18.511 | 0.000 | 24 | FAIL | xgboost-cuda direction/rank IC=0.03693 t=3.819 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2775 | 0.314 | -0.414 | -14.424 | 0.000 | 25 | FAIL | xgboost-cuda direction/rank IC=0.02456 t=2.51 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16068 | 0.262 | -0.520 | -34.972 | 0.000 | 26 | FAIL | xgboost-cuda direction/rank IC=0.07384 t=6.427 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5386 | 0.341 | -0.409 | -19.530 | 0.000 | 27 | FAIL | xgboost-cuda direction/rank IC=0.03088 t=3.146 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2631 | 0.333 | -0.395 | -14.061 | 0.000 | 28 | FAIL | xgboost-cuda direction/rank IC=0.02476 t=2.257 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 15994 | 0.263 | -0.514 | -33.885 | 0.000 | 29 | FAIL | xgboost-cuda direction/rank IC=0.07894 t=7.303 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5220 | 0.308 | -0.480 | -28.441 | 0.000 | 30 | FAIL | xgboost-cuda direction/rank IC=0.02967 t=2.962 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2625 | 0.326 | -0.394 | -13.575 | 0.000 | 31 | FAIL | xgboost-cuda direction/rank IC=0.01969 t=1.71 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16384 | 0.253 | -0.537 | -40.004 | 0.000 | 32 | FAIL | xgboost-cuda direction/rank IC=0.06898 t=5.098 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5254 | 0.335 | -0.402 | -18.468 | 0.000 | 33 | FAIL | xgboost-cuda direction/rank IC=0.02775 t=3.694 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2418 | 0.371 | -0.373 | -15.560 | 0.000 | 34 | FAIL | xgboost-cuda direction/rank IC=0.02043 t=2.262 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16503 | 0.252 | -0.552 | -48.148 | 0.000 | 35 | FAIL | xgboost-cuda direction/rank IC=0.07613 t=5.957 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5126 | 0.308 | -0.455 | -20.884 | 0.000 | 36 | FAIL | xgboost-cuda direction/rank IC=0.03073 t=4.117 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2339 | 0.314 | -0.390 | -13.120 | 0.000 | 37 | FAIL | xgboost-cuda direction/rank IC=0.02115 t=2.141 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 17558 | 0.273 | -0.510 | -42.633 | 0.000 | 38 | FAIL | xgboost-cuda direction/rank IC=0.06265 t=7.437 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16683 | 0.275 | -0.498 | -32.946 | 0.000 | 40 | FAIL | xgboost-cuda direction/rank IC=0.09411 t=7.543 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5288 | 0.268 | -0.509 | -20.220 | 0.000 | 41 | FAIL | xgboost-cuda direction/rank IC=0.0362 t=4.015 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2720 | 0.299 | -0.440 | -13.000 | 0.000 | 42 | FAIL | xgboost-cuda direction/rank IC=0.02632 t=2.261 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16604 | 0.267 | -0.522 | -36.537 | 0.000 | 43 | FAIL | xgboost-cuda direction/rank IC=0.09177 t=8.571 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5340 | 0.265 | -0.517 | -20.616 | 0.000 | 44 | FAIL | xgboost-cuda direction/rank IC=0.03538 t=3.643 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2670 | 0.280 | -0.465 | -13.582 | 0.000 | 45 | FAIL | xgboost-cuda direction/rank IC=0.01762 t=1.521 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 15489 | 0.243 | -0.565 | -51.814 | 0.000 | 46 | FAIL | xgboost-cuda direction/rank IC=0.08389 t=6.45 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5106 | 0.276 | -0.506 | -19.874 | 0.000 | 47 | FAIL | xgboost-cuda direction/rank IC=0.03429 t=3.149 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2872 | 0.276 | -0.483 | -19.863 | 0.000 | 48 | FAIL | xgboost-cuda direction/rank IC=0.02197 t=1.979 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 15915 | 0.260 | -0.518 | -32.460 | 0.000 | 49 | FAIL | xgboost-cuda direction/rank IC=0.08617 t=7.263 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5346 | 0.288 | -0.490 | -19.710 | 0.000 | 50 | FAIL | xgboost-cuda direction/rank IC=0.0335 t=3.297 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2898 | 0.294 | -0.463 | -17.101 | 0.000 | 51 | FAIL | xgboost-cuda direction/rank IC=0.01577 t=1.354 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16679 | 0.261 | -0.528 | -41.852 | 0.000 | 52 | FAIL | xgboost-cuda direction/rank IC=0.08014 t=6.27 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5260 | 0.310 | -0.451 | -19.694 | 0.000 | 53 | FAIL | xgboost-cuda direction/rank IC=0.03053 t=3.046 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2622 | 0.315 | -0.428 | -14.200 | 0.000 | 54 | FAIL | xgboost-cuda direction/rank IC=0.0241 t=2.507 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16212 | 0.243 | -0.562 | -53.837 | 0.000 | 55 | FAIL | xgboost-cuda direction/rank IC=0.08502 t=8.829 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5182 | 0.330 | -0.419 | -18.851 | 0.000 | 56 | FAIL | xgboost-cuda direction/rank IC=0.03693 t=3.819 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2775 | 0.314 | -0.414 | -14.753 | 0.000 | 57 | FAIL | xgboost-cuda direction/rank IC=0.02456 t=2.51 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16068 | 0.262 | -0.520 | -35.291 | 0.000 | 58 | FAIL | xgboost-cuda direction/rank IC=0.07384 t=6.427 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5386 | 0.341 | -0.409 | -19.839 | 0.000 | 59 | FAIL | xgboost-cuda direction/rank IC=0.03088 t=3.146 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2631 | 0.333 | -0.395 | -14.361 | 0.000 | 60 | FAIL | xgboost-cuda direction/rank IC=0.02476 t=2.257 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 15994 | 0.263 | -0.514 | -34.177 | 0.000 | 61 | FAIL | xgboost-cuda direction/rank IC=0.07894 t=7.303 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5220 | 0.308 | -0.480 | -28.725 | 0.000 | 62 | FAIL | xgboost-cuda direction/rank IC=0.02967 t=2.962 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2625 | 0.326 | -0.394 | -13.852 | 0.000 | 63 | FAIL | xgboost-cuda direction/rank IC=0.01969 t=1.71 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16384 | 0.253 | -0.537 | -40.274 | 0.000 | 64 | FAIL | xgboost-cuda direction/rank IC=0.06898 t=5.098 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5254 | 0.335 | -0.402 | -18.731 | 0.000 | 65 | FAIL | xgboost-cuda direction/rank IC=0.02775 t=3.694 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2418 | 0.371 | -0.373 | -15.816 | 0.000 | 66 | FAIL | xgboost-cuda direction/rank IC=0.02043 t=2.262 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16503 | 0.252 | -0.552 | -48.398 | 0.000 | 67 | FAIL | xgboost-cuda direction/rank IC=0.07613 t=5.957 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5126 | 0.308 | -0.455 | -21.128 | 0.000 | 68 | FAIL | xgboost-cuda direction/rank IC=0.03073 t=4.117 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2339 | 0.314 | -0.390 | -13.359 | 0.000 | 69 | FAIL | xgboost-cuda direction/rank IC=0.02115 t=2.141 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 17558 | 0.273 | -0.510 | -42.867 | 0.000 | 70 | FAIL | xgboost-cuda direction/rank IC=0.06265 t=7.437 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5247 | 0.325 | -0.430 | -19.604 | 0.000 | 71 | FAIL | xgboost-cuda direction/rank IC=0.01805 t=2.45 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2309 | 0.350 | -0.399 | -17.386 | 0.000 | 72 | FAIL | xgboost-cuda direction/rank IC=0.02185 t=1.992 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 5 | 16615 | 0.267 | -0.520 | -39.821 | 0.000 | 73 | FAIL | xgboost-cuda direction/rank IC=0.0632 t=5.92 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 15 | 5095 | 0.323 | -0.443 | -19.804 | 0.000 | 74 | FAIL | xgboost-cuda direction/rank IC=0.02263 t=2.562 |
| 2026-07-30 | B04b_xgb_sentiment_rank | 30 | 2272 | 0.319 | -0.401 | -15.667 | 0.000 | 75 | FAIL | xgboost-cuda direction/rank IC=0.01462 t=1.252 |
