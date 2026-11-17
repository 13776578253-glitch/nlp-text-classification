# EmotionLab Batch A results

Frozen manifest SHA-256: `255a9bfe53eea86f1339b2de4da46b96bfd9ab1594b723de5da11acf678c3289`

Each main row uses the seed 42 selected Run. The independent seed cohort is reported separately.

## goemotions

| Model | Validation Macro-F1 | Test Macro-F1 | Test Micro-F1 | Test Samples-F1 | Best epoch | Stop reason | Run ID | Training history | Selected parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.4778 | 0.4616 | 0.5392 | 0.5590 | — | single_fit | 20261117T160722-868a07d380 | [Curve](curves/goemotions-E1.png) | `{'c': 0.23345864076016243, 'class_weight': None, 'max_features': 100000, 'min_df': 2, 'ngram_range': [1, 1], 'sublinear_tf': True}` |
| E2 | 0.3938 | 0.3847 | 0.5217 | 0.5281 | 11 | epoch_cap | 20261117T162113-b964b3038d | [Curve](curves/goemotions-E2.png) | `{'dropout': 0.3, 'effective_batch_size': 64, 'embedding_dim': 50, 'epoch_cap': 12, 'grad_clip_norm': 1.0, 'learning_rate': 0.0010346264059718556, 'optimizer': 'adam', 'patience': 3, 'recurrent': {'hidden_size': 128}, 'textcnn': {'filters_per_size': 100, 'kernel_sizes': [3, 4, 5]}, 'weight_decay': 0.0}` |
| E3-RNN | 0.1976 | 0.2071 | 0.3892 | 0.4298 | 11 | epoch_cap | 20261117T155932-5a7fabb41f | [Curve](curves/goemotions-E3-RNN.png) | `{'dropout': 0.3, 'effective_batch_size': 64, 'embedding_dim': 50, 'epoch_cap': 12, 'grad_clip_norm': 1.0, 'learning_rate': 0.001, 'optimizer': 'adam', 'patience': 3, 'recurrent': {'hidden_size': 128}, 'textcnn': {'filters_per_size': 100, 'kernel_sizes': [3, 4, 5]}, 'weight_decay': 0.0}` |
| E3-LSTM | 0.3180 | 0.3116 | 0.4763 | 0.4972 | 12 | epoch_cap | 20261117T160026-136beb5c60 | [Curve](curves/goemotions-E3-LSTM.png) | `{'dropout': 0.3, 'effective_batch_size': 64, 'embedding_dim': 50, 'epoch_cap': 12, 'grad_clip_norm': 1.0, 'learning_rate': 0.001, 'optimizer': 'adam', 'patience': 3, 'recurrent': {'hidden_size': 128}, 'textcnn': {'filters_per_size': 100, 'kernel_sizes': [3, 4, 5]}, 'weight_decay': 0.0}` |
| E3-BiLSTM | 0.3882 | 0.3815 | 0.5134 | 0.5170 | 12 | epoch_cap | 20261117T160116-8e04380378 | [Curve](curves/goemotions-E3-BiLSTM.png) | `{'dropout': 0.3, 'effective_batch_size': 64, 'embedding_dim': 50, 'epoch_cap': 12, 'grad_clip_norm': 1.0, 'learning_rate': 0.001, 'optimizer': 'adam', 'patience': 3, 'recurrent': {'hidden_size': 128}, 'textcnn': {'filters_per_size': 100, 'kernel_sizes': [3, 4, 5]}, 'weight_decay': 0.0}` |
| E5-TUNE | 0.3949 | 0.3841 | 0.4927 | 0.5158 | 10 | epoch_cap | 20261117T163246-bd54996e97 | [Curve](curves/goemotions-E5-TUNE.png) | `{'dropout': 0.3, 'effective_batch_size': 64, 'embedding_dim': 50, 'epoch_cap': 12, 'grad_clip_norm': 1.0, 'learning_rate': 0.0010346264059718556, 'optimizer': 'adam', 'patience': 3, 'recurrent': {'hidden_size': 128}, 'textcnn': {'filters_per_size': 100, 'kernel_sizes': [3, 4, 5]}, 'weight_decay': 0.0}` |
| E5-FROZEN | 0.2698 | 0.2569 | 0.3870 | 0.4169 | 10 | epoch_cap | 20261117T163326-c4703be677 | [Curve](curves/goemotions-E5-FROZEN.png) | `{'dropout': 0.3, 'effective_batch_size': 64, 'embedding_dim': 50, 'epoch_cap': 12, 'grad_clip_norm': 1.0, 'learning_rate': 0.0010346264059718556, 'optimizer': 'adam', 'patience': 3, 'recurrent': {'hidden_size': 128}, 'textcnn': {'filters_per_size': 100, 'kernel_sizes': [3, 4, 5]}, 'weight_decay': 0.0}` |

### Embedding ablation

| Model | Test Macro-F1 | Vocabulary coverage | Token coverage | Run ID |
| --- | --- | --- | --- | --- |
| E2 | 0.3847 | — | — | 20261117T162113-b964b3038d |
| E5-TUNE | 0.3841 | 0.5015918322538149 | 0.9795059766711206 | 20261117T163246-bd54996e97 |
| E5-FROZEN | 0.2569 | 0.5015918322538149 | 0.9795059766711206 | 20261117T163326-c4703be677 |

## movie_reviews_zh

| Model | Validation Macro-F1 | Test Macro-F1 | Test Micro-F1 | Test accuracy | Test positive F1 | Best epoch | Stop reason | Run ID | Training history | Selected parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.8581 | 0.8371 | 0.8374 | 0.8374 | 0.8438 | — | single_fit | 20261117T163440-c6504c2d90 | [Curve](curves/movie_reviews_zh-E1.png) | `{'c': 0.6467386144189291, 'class_weight': None, 'max_features': 100000, 'min_df': 2, 'ngram_range': [1, 2], 'sublinear_tf': True}` |
| E2 | 0.8294 | 0.8075 | 0.8076 | 0.8076 | 0.8117 | 7 | early_stopping | 20261117T163624-36ca4b1c73 | [Curve](curves/movie_reviews_zh-E2.png) | `{'dropout': 0.1, 'effective_batch_size': 64, 'embedding_dim': 50, 'epoch_cap': 12, 'grad_clip_norm': 1.0, 'learning_rate': 0.0016178172626630791, 'optimizer': 'adam', 'patience': 3, 'recurrent': {'hidden_size': 128}, 'textcnn': {'filters_per_size': 100, 'kernel_sizes': [3, 4, 5]}, 'weight_decay': 0.0001}` |
| E3-RNN | 0.8281 | 0.8202 | 0.8211 | 0.8211 | 0.8333 | 8 | early_stopping | 20261117T163840-cdd3c0141e | [Curve](curves/movie_reviews_zh-E3-RNN.png) | `{'dropout': 0.1, 'effective_batch_size': 64, 'embedding_dim': 50, 'epoch_cap': 12, 'grad_clip_norm': 1.0, 'learning_rate': 0.0010100610125437241, 'optimizer': 'adam', 'patience': 3, 'recurrent': {'hidden_size': 128}, 'textcnn': {'filters_per_size': 100, 'kernel_sizes': [3, 4, 5]}, 'weight_decay': 0.0001}` |
| E3-LSTM | 0.8348 | 0.8453 | 0.8455 | 0.8455 | 0.8512 | 5 | early_stopping | 20261117T164106-18b651db68 | [Curve](curves/movie_reviews_zh-E3-LSTM.png) | `{'dropout': 0.1, 'effective_batch_size': 64, 'embedding_dim': 50, 'epoch_cap': 12, 'grad_clip_norm': 1.0, 'learning_rate': 0.0013812860458108246, 'optimizer': 'adam', 'patience': 3, 'recurrent': {'hidden_size': 128}, 'textcnn': {'filters_per_size': 100, 'kernel_sizes': [3, 4, 5]}, 'weight_decay': 0.0001}` |
| E3-BiLSTM | 0.8346 | 0.8313 | 0.8320 | 0.8320 | 0.8418 | 5 | early_stopping | 20261117T164353-1b486fe355 | [Curve](curves/movie_reviews_zh-E3-BiLSTM.png) | `{'dropout': 0.1, 'effective_batch_size': 64, 'embedding_dim': 50, 'epoch_cap': 12, 'grad_clip_norm': 1.0, 'learning_rate': 0.0013812860458108246, 'optimizer': 'adam', 'patience': 3, 'recurrent': {'hidden_size': 128}, 'textcnn': {'filters_per_size': 100, 'kernel_sizes': [3, 4, 5]}, 'weight_decay': 0.0001}` |
| E5-TUNE | 0.8349 | 0.8428 | 0.8428 | 0.8428 | 0.8415 | 4 | early_stopping | 20261117T164438-8338345ac2 | [Curve](curves/movie_reviews_zh-E5-TUNE.png) | `{'dropout': 0.1, 'effective_batch_size': 64, 'embedding_dim': 50, 'epoch_cap': 12, 'grad_clip_norm': 1.0, 'learning_rate': 0.0016178172626630791, 'optimizer': 'adam', 'patience': 3, 'recurrent': {'hidden_size': 128}, 'textcnn': {'filters_per_size': 100, 'kernel_sizes': [3, 4, 5]}, 'weight_decay': 0.0001}` |
| E5-FROZEN | 0.7879 | 0.7739 | 0.7751 | 0.7751 | 0.7899 | 3 | early_stopping | 20261117T164453-951f19e435 | [Curve](curves/movie_reviews_zh-E5-FROZEN.png) | `{'dropout': 0.1, 'effective_batch_size': 64, 'embedding_dim': 50, 'epoch_cap': 12, 'grad_clip_norm': 1.0, 'learning_rate': 0.0016178172626630791, 'optimizer': 'adam', 'patience': 3, 'recurrent': {'hidden_size': 128}, 'textcnn': {'filters_per_size': 100, 'kernel_sizes': [3, 4, 5]}, 'weight_decay': 0.0001}` |

### Embedding ablation

| Model | Test Macro-F1 | Vocabulary coverage | Token coverage | Run ID |
| --- | --- | --- | --- | --- |
| E2 | 0.8075 | — | — | 20261117T163624-36ca4b1c73 |
| E5-TUNE | 0.8428 | 0.8300301876886731 | 0.9717884937444586 | 20261117T164438-8338345ac2 |
| E5-FROZEN | 0.7739 | 0.8300301876886731 | 0.9717884937444586 | 20261117T164453-951f19e435 |

## Hyperparameter search

| Study | Trial | Status | Validation Macro-F1 | Seconds | Stop reason | Run | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| goemotions-E1 | 0 | PASS | 0.4382 | 8.01 | val_macro_f1=0.43816 | 20261117T160657-1deb3d45eb | `{'c': 0.13292918943162169, 'min_df': 2, 'ngram_range': [1, 1], 'class_weight': 'balanced'}` |
| goemotions-E1 | 1 | PASS | 0.3938 | 8.05 | val_macro_f1=0.39376 | 20261117T160706-3ea7a439c2 | `{'c': 0.6358358856676253, 'min_df': 5, 'ngram_range': [1, 1], 'class_weight': 'balanced'}` |
| goemotions-E1 | 2 | PASS | 0.4419 | 7.88 | val_macro_f1=0.44189 | 20261117T160714-00bf23e7b6 | `{'c': 0.08179499475211674, 'min_df': 2, 'ngram_range': [1, 1], 'class_weight': 'balanced'}` |
| goemotions-E1 | 3 | PASS | 0.4778 | 4.96 | val_macro_f1=0.47776 | 20261117T160722-868a07d380 | `{'c': 0.23345864076016243, 'min_df': 2, 'ngram_range': [1, 1], 'class_weight': None}` |
| goemotions-E1 | 4 | PASS | 0.4211 | 3.26 | val_macro_f1=0.42107 | 20261117T160727-995b0f7b21 | `{'c': 0.015673095467235415, 'min_df': 3, 'ngram_range': [1, 1], 'class_weight': None}` |
| goemotions-E1 | 5 | PASS | 0.4051 | 12.99 | val_macro_f1=0.40514 | 20261117T160731-ae5eb666cc | `{'c': 5.0729570336878345, 'min_df': 2, 'ngram_range': [1, 2], 'class_weight': None}` |
| goemotions-E1 | 6 | PASS | 0.3981 | 7.17 | val_macro_f1=0.39809 | 20261117T160744-6ffd9c7ad3 | `{'c': 2.768007873337228, 'min_df': 2, 'ngram_range': [1, 1], 'class_weight': None}` |
| goemotions-E1 | 7 | PASS | 0.4150 | 3.77 | val_macro_f1=0.41504 | 20261117T160752-46bc92bd2c | `{'c': 0.01452181284656382, 'min_df': 2, 'ngram_range': [1, 1], 'class_weight': None}` |
| goemotions-E1 | 8 | PASS | 0.4214 | 5.44 | val_macro_f1=0.42139 | 20261117T160756-99d0c0268c | `{'c': 0.02380449190659745, 'min_df': 5, 'ngram_range': [1, 2], 'class_weight': None}` |
| goemotions-E1 | 9 | PASS | 0.4769 | 7.24 | val_macro_f1=0.47692 | 20261117T160801-837e9b934d | `{'c': 0.27242832910714504, 'min_df': 3, 'ngram_range': [1, 2], 'class_weight': None}` |
| goemotions-E1 | 10 | PASS | 0.3307 | 40.25 | val_macro_f1=0.33068 | 20261117T160809-583e75b7b6 | `{'c': 9.275324466609787, 'min_df': 3, 'ngram_range': [1, 1], 'class_weight': 'balanced'}` |
| goemotions-E1 | 11 | PASS | 0.4659 | 8.79 | val_macro_f1=0.46594 | 20261117T160850-5bc01d4db3 | `{'c': 0.5796548071092072, 'min_df': 3, 'ngram_range': [1, 2], 'class_weight': None}` |
| goemotions-E1 | 12 | PASS | 0.4769 | 4.53 | val_macro_f1=0.47686 | 20261117T160859-81ba6c5ca5 | `{'c': 0.19844562171809382, 'min_df': 3, 'ngram_range': [1, 1], 'class_weight': None}` |
| goemotions-E1 | 13 | PASS | 0.4611 | 6.16 | val_macro_f1=0.46107 | 20261117T160903-3bb7e98694 | `{'c': 0.06957495919434674, 'min_df': 2, 'ngram_range': [1, 2], 'class_weight': None}` |
| goemotions-E1 | 14 | PASS | 0.4451 | 16.67 | val_macro_f1=0.44514 | 20261117T160910-ff548b982c | `{'c': 0.41577686510837225, 'min_df': 3, 'ngram_range': [1, 2], 'class_weight': 'balanced'}` |
| goemotions-E1 | 15 | PASS | 0.4372 | 7.20 | val_macro_f1=0.43723 | 20261117T160927-28a8288b71 | `{'c': 0.0314688951902829, 'min_df': 3, 'ngram_range': [1, 2], 'class_weight': None}` |
| goemotions-E2 | 0 | PASS | 0.3455 | 52.96 | val_macro_f1=0.34546 | 20261117T160934-bda1eb9c3b | `{'learning_rate': 0.0005514718587869892, 'weight_decay': 0.0, 'dropout': 0.5}` |
| goemotions-E2 | 1 | PASS | 0.3335 | 52.47 | val_macro_f1=0.33355 | 20261117T161028-38116a079d | `{'learning_rate': 0.001018582256686764, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| goemotions-E2 | 2 | PASS | 0.1060 | 23.89 | val_macro_f1=0.10599 | 20261117T161120-1efce2e5e8 | `{'learning_rate': 0.00045587590475301384, 'weight_decay': 0.001, 'dropout': 0.5}` |
| goemotions-E2 | 3 | PASS | 0.3714 | 4.68 | val_macro_f1=0.37139 | 20261117T161753-445e173c1f | `{'learning_rate': 0.0006877172495876584, 'weight_decay': 0.0, 'dropout': 0.3}` |
| goemotions-E2 | 4 | PASS | 0.3938 | 25.94 | val_macro_f1=0.39380 | 20261117T162113-b964b3038d | `{'learning_rate': 0.0010346264059718556, 'weight_decay': 0.0, 'dropout': 0.3}` |
| goemotions-E2 | 5 | PRUNED | — | 16.83 | Optuna median pruning at epoch 6 | 20261117T162140-8ff0358cf3 | `{'learning_rate': 0.00020439036385964718, 'weight_decay': 1e-05, 'dropout': 0.3}` |
| goemotions-E3-RNN | 0 | PASS | 0.1513 | 37.61 | val_macro_f1=0.15131 | 20261117T162157-f0adb3c6a6 | `{'learning_rate': 0.0003071057367777374, 'weight_decay': 0.0, 'dropout': 0.5}` |
| goemotions-E3-RNN | 1 | PASS | 0.1692 | 43.28 | val_macro_f1=0.16922 | 20261117T162235-22bb3573c9 | `{'learning_rate': 0.0006054365855469249, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| goemotions-E3-RNN | 2 | PASS | 0.0691 | 36.43 | val_macro_f1=0.06913 | 20261117T162318-e538c8f4fb | `{'learning_rate': 0.0002487873441981445, 'weight_decay': 0.001, 'dropout': 0.5}` |
| goemotions-E3-RNN | 3 | PASS | 0.1690 | 38.25 | val_macro_f1=0.16898 | 20261117T162355-c08f8cb52d | `{'learning_rate': 0.0003920673972242139, 'weight_decay': 0.0, 'dropout': 0.3}` |
| goemotions-E3-RNN | 4 | PASS | 0.1182 | 26.40 | val_macro_f1=0.11817 | 20261117T162433-2f7689f0ca | `{'learning_rate': 0.0010100610125437241, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| goemotions-E3-RNN | 5 | PRUNED | — | 18.73 | Optuna median pruning at epoch 6 | 20261117T162500-60beaf7fbd | `{'learning_rate': 0.00010737587043362551, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| goemotions-E3-LSTM | 0 | PASS | 0.1610 | 38.40 | val_macro_f1=0.16098 | 20261117T162519-74472fb774 | `{'learning_rate': 0.0003574712922600243, 'weight_decay': 0.0, 'dropout': 0.5}` |
| goemotions-E3-LSTM | 1 | PASS | 0.2081 | 39.35 | val_macro_f1=0.20810 | 20261117T162558-a98d0776ae | `{'learning_rate': 0.0007725378389307352, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| goemotions-E3-LSTM | 2 | PASS | 0.0653 | 14.61 | val_macro_f1=0.06532 | 20261117T162637-d18c2a48ed | `{'learning_rate': 0.0002814509271606064, 'weight_decay': 0.001, 'dropout': 0.5}` |
| goemotions-E3-LSTM | 3 | PASS | 0.2201 | 40.65 | val_macro_f1=0.22011 | 20261117T162652-9232e205a1 | `{'learning_rate': 0.000471705203762518, 'weight_decay': 0.0, 'dropout': 0.3}` |
| goemotions-E3-LSTM | 4 | PASS | 0.2771 | 38.40 | val_macro_f1=0.27709 | 20261117T162733-62c795e4c3 | `{'learning_rate': 0.0007878517696049197, 'weight_decay': 0.0, 'dropout': 0.3}` |
| goemotions-E3-LSTM | 5 | PASS | 0.0657 | 14.62 | val_macro_f1=0.06572 | 20261117T162812-4ade04ea10 | `{'learning_rate': 0.002389619932866104, 'weight_decay': 0.001, 'dropout': 0.3}` |
| goemotions-E3-BiLSTM | 0 | PASS | 0.2746 | 51.40 | val_macro_f1=0.27459 | 20261117T162827-0769cddaa5 | `{'learning_rate': 0.0003574712922600243, 'weight_decay': 0.0, 'dropout': 0.5}` |
| goemotions-E3-BiLSTM | 1 | PASS | 0.2706 | 51.75 | val_macro_f1=0.27061 | 20261117T162919-5de08a1b91 | `{'learning_rate': 0.0007725378389307352, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| goemotions-E3-BiLSTM | 2 | PASS | 0.0671 | 19.30 | val_macro_f1=0.06707 | 20261117T163011-d90e838576 | `{'learning_rate': 0.0002814509271606064, 'weight_decay': 0.001, 'dropout': 0.5}` |
| goemotions-E3-BiLSTM | 3 | PASS | 0.3294 | 56.72 | val_macro_f1=0.32936 | 20261117T163030-05f3a68ad4 | `{'learning_rate': 0.000471705203762518, 'weight_decay': 0.0, 'dropout': 0.3}` |
| goemotions-E3-BiLSTM | 4 | PASS | 0.3810 | 58.01 | val_macro_f1=0.38102 | 20261117T163127-4e6800dd6d | `{'learning_rate': 0.0007878517696049197, 'weight_decay': 0.0, 'dropout': 0.3}` |
| goemotions-E3-BiLSTM | 5 | PASS | 0.0676 | 19.67 | val_macro_f1=0.06760 | 20261117T163226-535c801732 | `{'learning_rate': 0.002389619932866104, 'weight_decay': 0.001, 'dropout': 0.3}` |
| movie_reviews_zh-E1 | 0 | PASS | 0.8499 | 1.14 | val_macro_f1=0.84986 | 20261117T163408-43a942fe14 | `{'c': 0.13292918943162169, 'min_df': 2, 'ngram_range': [1, 1], 'class_weight': 'balanced'}` |
| movie_reviews_zh-E1 | 1 | PASS | 0.8442 | 0.89 | val_macro_f1=0.84417 | 20261117T163409-1d94f85356 | `{'c': 0.6358358856676253, 'min_df': 5, 'ngram_range': [1, 1], 'class_weight': 'balanced'}` |
| movie_reviews_zh-E1 | 2 | PASS | 0.8465 | 1.05 | val_macro_f1=0.84649 | 20261117T163410-2816c391f4 | `{'c': 0.08179499475211674, 'min_df': 2, 'ngram_range': [1, 1], 'class_weight': 'balanced'}` |
| movie_reviews_zh-E1 | 3 | PASS | 0.8500 | 1.18 | val_macro_f1=0.85005 | 20261117T163411-25e2ee970c | `{'c': 0.23345864076016243, 'min_df': 2, 'ngram_range': [1, 1], 'class_weight': None}` |
| movie_reviews_zh-E1 | 4 | PASS | 0.8243 | 0.96 | val_macro_f1=0.82430 | 20261117T163413-b4a18000ca | `{'c': 0.015673095467235415, 'min_df': 3, 'ngram_range': [1, 1], 'class_weight': None}` |
| movie_reviews_zh-E1 | 5 | PASS | 0.8522 | 4.45 | val_macro_f1=0.85221 | 20261117T163414-62c1c4d577 | `{'c': 5.0729570336878345, 'min_df': 2, 'ngram_range': [1, 2], 'class_weight': None}` |
| movie_reviews_zh-E1 | 6 | PASS | 0.8524 | 4.39 | val_macro_f1=0.85239 | 20261117T163419-fc718fbad3 | `{'c': 5.2213675170252465, 'min_df': 2, 'ngram_range': [1, 2], 'class_weight': None}` |
| movie_reviews_zh-E1 | 7 | PASS | 0.8220 | 4.08 | val_macro_f1=0.82199 | 20261117T163423-0e7078c2cc | `{'c': 0.015424898975561959, 'min_df': 2, 'ngram_range': [1, 2], 'class_weight': None}` |
| movie_reviews_zh-E1 | 8 | PASS | 0.8482 | 3.92 | val_macro_f1=0.84815 | 20261117T163427-dc6ed7fdee | `{'c': 4.538494101083456, 'min_df': 3, 'ngram_range': [1, 2], 'class_weight': 'balanced'}` |
| movie_reviews_zh-E1 | 9 | PASS | 0.8522 | 3.54 | val_macro_f1=0.85218 | 20261117T163432-8695b6566b | `{'c': 0.14780104887814832, 'min_df': 5, 'ngram_range': [1, 2], 'class_weight': None}` |
| movie_reviews_zh-E1 | 10 | PASS | 0.8573 | 4.17 | val_macro_f1=0.85734 | 20261117T163435-6b3b2cdd8a | `{'c': 0.7858746119481282, 'min_df': 2, 'ngram_range': [1, 2], 'class_weight': None}` |
| movie_reviews_zh-E1 | 11 | PASS | 0.8581 | 4.02 | val_macro_f1=0.85805 | 20261117T163440-c6504c2d90 | `{'c': 0.6467386144189291, 'min_df': 2, 'ngram_range': [1, 2], 'class_weight': None}` |
| movie_reviews_zh-E1 | 12 | PASS | 0.8573 | 4.07 | val_macro_f1=0.85734 | 20261117T163444-908690b05e | `{'c': 0.4226074144109217, 'min_df': 2, 'ngram_range': [1, 2], 'class_weight': None}` |
| movie_reviews_zh-E1 | 13 | PASS | 0.8515 | 3.10 | val_macro_f1=0.85148 | 20261117T163448-32629ba996 | `{'c': 0.9798072564932205, 'min_df': 5, 'ngram_range': [1, 2], 'class_weight': None}` |
| movie_reviews_zh-E1 | 14 | PASS | 0.8460 | 4.11 | val_macro_f1=0.84596 | 20261117T163451-0c2614f867 | `{'c': 0.08983862502145487, 'min_df': 2, 'ngram_range': [1, 2], 'class_weight': None}` |
| movie_reviews_zh-E1 | 15 | PASS | 0.8377 | 1.17 | val_macro_f1=0.83766 | 20261117T163456-7c354fb438 | `{'c': 1.5903523457499478, 'min_df': 2, 'ngram_range': [1, 1], 'class_weight': None}` |
| movie_reviews_zh-E2 | 0 | PASS | 0.8216 | 21.32 | val_macro_f1=0.82164 | 20261117T163457-7df912509a | `{'learning_rate': 0.0005514718587869892, 'weight_decay': 0.0, 'dropout': 0.5}` |
| movie_reviews_zh-E2 | 1 | PASS | 0.8265 | 21.50 | val_macro_f1=0.82654 | 20261117T163519-9fc42eb710 | `{'learning_rate': 0.001018582256686764, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| movie_reviews_zh-E2 | 2 | PASS | 0.7838 | 21.32 | val_macro_f1=0.78384 | 20261117T163541-8ca4ed726e | `{'learning_rate': 0.00045587590475301384, 'weight_decay': 0.001, 'dropout': 0.5}` |
| movie_reviews_zh-E2 | 3 | PASS | 0.8202 | 21.12 | val_macro_f1=0.82022 | 20261117T163602-a9ac90f562 | `{'learning_rate': 0.0006877172495876584, 'weight_decay': 0.0, 'dropout': 0.3}` |
| movie_reviews_zh-E2 | 4 | PASS | 0.8294 | 18.29 | val_macro_f1=0.82937 | 20261117T163624-36ca4b1c73 | `{'learning_rate': 0.0016178172626630791, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| movie_reviews_zh-E2 | 5 | PRUNED | — | 11.39 | Optuna median pruning at epoch 6 | 20261117T163642-6f943916f1 | `{'learning_rate': 0.00021333959964409978, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| movie_reviews_zh-E3-RNN | 0 | PASS | 0.7987 | 26.72 | val_macro_f1=0.79869 | 20261117T163654-85df859366 | `{'learning_rate': 0.0003071057367777374, 'weight_decay': 0.0, 'dropout': 0.5}` |
| movie_reviews_zh-E3-RNN | 1 | PASS | 0.8149 | 25.92 | val_macro_f1=0.81487 | 20261117T163721-97b2bc833e | `{'learning_rate': 0.0006054365855469249, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| movie_reviews_zh-E3-RNN | 2 | PASS | 0.7013 | 26.46 | val_macro_f1=0.70130 | 20261117T163747-42093c2c57 | `{'learning_rate': 0.0002487873441981445, 'weight_decay': 0.001, 'dropout': 0.5}` |
| movie_reviews_zh-E3-RNN | 3 | PASS | 0.8022 | 26.38 | val_macro_f1=0.80220 | 20261117T163814-e201930b41 | `{'learning_rate': 0.0003920673972242139, 'weight_decay': 0.0, 'dropout': 0.3}` |
| movie_reviews_zh-E3-RNN | 4 | PASS | 0.8281 | 25.92 | val_macro_f1=0.82811 | 20261117T163840-cdd3c0141e | `{'learning_rate': 0.0010100610125437241, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| movie_reviews_zh-E3-RNN | 5 | PRUNED | — | 12.85 | Optuna median pruning at epoch 6 | 20261117T163907-7128c4eb7c | `{'learning_rate': 0.00010740398311009175, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| movie_reviews_zh-E3-LSTM | 0 | PASS | 0.8175 | 28.32 | val_macro_f1=0.81751 | 20261117T163920-40e265a78a | `{'learning_rate': 0.0003574712922600243, 'weight_decay': 0.0, 'dropout': 0.5}` |
| movie_reviews_zh-E3-LSTM | 1 | PASS | 0.8326 | 28.19 | val_macro_f1=0.83261 | 20261117T163948-ad95becaf3 | `{'learning_rate': 0.0007725378389307352, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| movie_reviews_zh-E3-LSTM | 2 | PASS | 0.7536 | 23.20 | val_macro_f1=0.75359 | 20261117T164017-20b1a0781a | `{'learning_rate': 0.0002814509271606064, 'weight_decay': 0.001, 'dropout': 0.5}` |
| movie_reviews_zh-E3-LSTM | 3 | PASS | 0.8177 | 25.34 | val_macro_f1=0.81772 | 20261117T164040-400bdf7fc2 | `{'learning_rate': 0.000471705203762518, 'weight_decay': 0.0, 'dropout': 0.3}` |
| movie_reviews_zh-E3-LSTM | 4 | PASS | 0.8348 | 18.67 | val_macro_f1=0.83478 | 20261117T164106-18b651db68 | `{'learning_rate': 0.0013812860458108246, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| movie_reviews_zh-E3-LSTM | 5 | PRUNED | — | 13.68 | Optuna median pruning at epoch 6 | 20261117T164125-6609b60d6e | `{'learning_rate': 0.00010844734445826808, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| movie_reviews_zh-E3-BiLSTM | 0 | PASS | 0.8149 | 35.52 | val_macro_f1=0.81488 | 20261117T164139-f359dee8d5 | `{'learning_rate': 0.0003574712922600243, 'weight_decay': 0.0, 'dropout': 0.5}` |
| movie_reviews_zh-E3-BiLSTM | 1 | PASS | 0.8344 | 34.30 | val_macro_f1=0.83442 | 20261117T164215-0fa6974e8a | `{'learning_rate': 0.0007725378389307352, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| movie_reviews_zh-E3-BiLSTM | 2 | PASS | 0.7566 | 34.73 | val_macro_f1=0.75662 | 20261117T164250-7c248843bc | `{'learning_rate': 0.0002814509271606064, 'weight_decay': 0.001, 'dropout': 0.5}` |
| movie_reviews_zh-E3-BiLSTM | 3 | PASS | 0.8142 | 28.01 | val_macro_f1=0.81415 | 20261117T164325-d4251b719d | `{'learning_rate': 0.000471705203762518, 'weight_decay': 0.0, 'dropout': 0.3}` |
| movie_reviews_zh-E3-BiLSTM | 4 | PASS | 0.8346 | 23.89 | val_macro_f1=0.83456 | 20261117T164353-1b486fe355 | `{'learning_rate': 0.0013812860458108246, 'weight_decay': 0.0001, 'dropout': 0.1}` |
| movie_reviews_zh-E3-BiLSTM | 5 | PRUNED | — | 19.14 | Optuna median pruning at epoch 6 | 20261117T164417-14566da7c7 | `{'learning_rate': 0.00010844734445826808, 'weight_decay': 0.0001, 'dropout': 0.1}` |

## Independent seed replications

| Dataset | Model | Validation Macro-F1 by seed | Test Macro-F1 by seed | n | Validation mean ± SD | Test mean ± SD |
| --- | --- | --- | --- | --- | --- | --- |
| goemotions | E1 | 42: 0.4778, 123: 0.4778, 2026: 0.4778 | 42: 0.4616, 123: 0.4616, 2026: 0.4616 | 3 | 0.4778 ± 0.0000 | 0.4616 ± 0.0000 |
| goemotions | E2 | 42: 0.3938, 123: 0.3906, 2026: 0.3932 | 42: 0.3847, 123: 0.3832, 2026: 0.3833 | 3 | 0.3925 ± 0.0017 | 0.3837 ± 0.0008 |
| goemotions | E3-BiLSTM | 42: 0.3882, 123: 0.4023, 2026: 0.3778 | 42: 0.3815, 123: 0.3750, 2026: 0.3540 | 3 | 0.3894 ± 0.0123 | 0.3702 ± 0.0143 |
| goemotions | E5-FROZEN | 42: 0.2698, 123: 0.2724, 2026: 0.2790 | 42: 0.2569, 123: 0.2642, 2026: 0.2728 | 3 | 0.2737 ± 0.0047 | 0.2647 ± 0.0080 |
| goemotions | E5-TUNE | 42: 0.3949, 123: 0.3738, 2026: 0.3863 | 42: 0.3841, 123: 0.3690, 2026: 0.3756 | 3 | 0.3850 ± 0.0106 | 0.3763 ± 0.0076 |
| movie_reviews_zh | E1 | 42: 0.8581, 123: 0.8581, 2026: 0.8581 | 42: 0.8371, 123: 0.8371, 2026: 0.8371 | 3 | 0.8581 ± 0.0000 | 0.8371 ± 0.0000 |
| movie_reviews_zh | E2 | 42: 0.8294, 123: 0.8231, 2026: 0.8284 | 42: 0.8075, 123: 0.8450, 2026: 0.8427 | 3 | 0.8269 ± 0.0034 | 0.8317 ± 0.0210 |
| movie_reviews_zh | E3-LSTM | 42: 0.8348, 123: 0.8314, 2026: 0.8305 | 42: 0.8453, 123: 0.8265, 2026: 0.8317 | 3 | 0.8322 ± 0.0023 | 0.8345 ± 0.0097 |
| movie_reviews_zh | E5-FROZEN | 42: 0.7879, 123: 0.7838, 2026: 0.7953 | 42: 0.7739, 123: 0.7994, 2026: 0.8020 | 3 | 0.7890 ± 0.0059 | 0.7918 ± 0.0155 |
| movie_reviews_zh | E5-TUNE | 42: 0.8349, 123: 0.8369, 2026: 0.8431 | 42: 0.8428, 123: 0.8453, 2026: 0.8399 | 3 | 0.8383 ± 0.0042 | 0.8427 ± 0.0027 |

### Additional validation-selected runs

| Run key | Run ID | Test Macro-F1 |
| --- | --- | --- |
| movie_reviews_zh/E3-BiLSTM/42 | 20261117T165716-66fc569677 | 0.8313 |
| movie_reviews_zh/E3-BiLSTM/123 | 20261117T165739-89262b24b0 | 0.8401 |
| movie_reviews_zh/E3-BiLSTM/2026 | 20261117T165803-4f35577931 | 0.8231 |

## Chinese paired Bootstrap

10,000 label-stratified paired resamples.

| Model A | Model B | Macro-F1 A − B | 95% percentile CI | n |
| --- | --- | --- | --- | --- |
| E2 | E1 | -0.0296 | [-0.0624, 0.0028] | 369 |
| E3-LSTM | E2 | 0.0378 | [0.0053, 0.0705] | 369 |
| E5-TUNE | E2 | 0.0353 | [0.0027, 0.0705] | 369 |
| E5-FROZEN | E2 | -0.0336 | [-0.0787, 0.0126] | 369 |
| E5-TUNE | E5-FROZEN | 0.0689 | [0.0300, 0.1093] | 369 |

## GoEmotions low-support labels

Five lowest train support labels and all labels meeting the rare-label rule. 0 labels meet the frozen train support cutoff (<50). Confidence intervals apply to rare labels with test support of at least 10.

| Model | Label | Train support | Test support | F1 | 95% CI |
| --- | --- | --- | --- | --- | --- |
| E1 | embarrassment | 302 | 37 | 0.4127 | — |
| E1 | grief | 77 | 6 | 0.4444 | — |
| E1 | nervousness | 164 | 23 | 0.3182 | — |
| E1 | pride | 111 | 16 | 0.4800 | — |
| E1 | relief | 153 | 11 | 0.1667 | — |
| E2 | embarrassment | 302 | 37 | 0.0909 | — |
| E2 | grief | 77 | 6 | 0.0000 | — |
| E2 | nervousness | 164 | 23 | 0.1379 | — |
| E2 | pride | 111 | 16 | 0.2857 | — |
| E2 | relief | 153 | 11 | 0.0000 | — |
| E3-RNN | embarrassment | 302 | 37 | 0.0000 | — |
| E3-RNN | grief | 77 | 6 | 0.0000 | — |
| E3-RNN | nervousness | 164 | 23 | 0.0000 | — |
| E3-RNN | pride | 111 | 16 | 0.0000 | — |
| E3-RNN | relief | 153 | 11 | 0.0000 | — |
| E3-LSTM | embarrassment | 302 | 37 | 0.0000 | — |
| E3-LSTM | grief | 77 | 6 | 0.0000 | — |
| E3-LSTM | nervousness | 164 | 23 | 0.0000 | — |
| E3-LSTM | pride | 111 | 16 | 0.0000 | — |
| E3-LSTM | relief | 153 | 11 | 0.0000 | — |
| E3-BiLSTM | embarrassment | 302 | 37 | 0.1311 | — |
| E3-BiLSTM | grief | 77 | 6 | 0.0000 | — |
| E3-BiLSTM | nervousness | 164 | 23 | 0.2069 | — |
| E3-BiLSTM | pride | 111 | 16 | 0.0000 | — |
| E3-BiLSTM | relief | 153 | 11 | 0.0000 | — |
| E5-TUNE | embarrassment | 302 | 37 | 0.1818 | — |
| E5-TUNE | grief | 77 | 6 | 0.0000 | — |
| E5-TUNE | nervousness | 164 | 23 | 0.1091 | — |
| E5-TUNE | pride | 111 | 16 | 0.2000 | — |
| E5-TUNE | relief | 153 | 11 | 0.0909 | — |
| E5-FROZEN | embarrassment | 302 | 37 | 0.0000 | — |
| E5-FROZEN | grief | 77 | 6 | 0.0000 | — |
| E5-FROZEN | nervousness | 164 | 23 | 0.0000 | — |
| E5-FROZEN | pride | 111 | 16 | 0.0000 | — |
| E5-FROZEN | relief | 153 | 11 | 0.2105 | — |

## Failed, pruned, or interrupted attempts

| Task | Run | Status | Reason |
| --- | --- | --- | --- |
| hpo/goemotions/E2/42/3 | artifacts\runs\goemotions\E2\20261117T161144-21043ec89d | INCOMPLETE | Operator paused the batch after workstation memory pressure was reported |
| hpo/goemotions/E2/42/4 | artifacts\runs\goemotions\E2\20261117T161758-2a07c98c25 | INCOMPLETE | Stopped the short guarded resume check to keep the queue paused during verification |
| hpo/goemotions/E2/42/5 | artifacts\runs\goemotions\E2\20261117T162140-8ff0358cf3 | PRUNED | Optuna median pruning at epoch 6 |
| hpo/goemotions/E3-RNN/42/5 | artifacts\runs\goemotions\E3-RNN\20261117T162500-60beaf7fbd | PRUNED | Optuna median pruning at epoch 6 |
| hpo/movie_reviews_zh/E2/42/5 | artifacts\runs\movie_reviews_zh\E2\20261117T163642-6f943916f1 | PRUNED | Optuna median pruning at epoch 6 |
| hpo/movie_reviews_zh/E3-RNN/42/5 | artifacts\runs\movie_reviews_zh\E3-RNN\20261117T163907-7128c4eb7c | PRUNED | Optuna median pruning at epoch 6 |
| hpo/movie_reviews_zh/E3-LSTM/42/5 | artifacts\runs\movie_reviews_zh\E3-LSTM\20261117T164125-6609b60d6e | PRUNED | Optuna median pruning at epoch 6 |
| hpo/movie_reviews_zh/E3-BiLSTM/42/5 | artifacts\runs\movie_reviews_zh\E3-BiLSTM\20261117T164417-14566da7c7 | PRUNED | Optuna median pruning at epoch 6 |

## Performance review

| Dataset | Model | Flags | Train Macro-F1 | Validation Macro-F1 |
| --- | --- | --- | --- | --- |
| goemotions | E1 |  | 0.6943 | 0.4778 |
| goemotions | E2 |  | 0.5868 | 0.3938 |
| goemotions | E3-RNN | underfitting | 0.2159 | 0.1976 |
| goemotions | E3-LSTM |  | 0.3912 | 0.3180 |
| goemotions | E3-BiLSTM |  | 0.5506 | 0.3882 |
| goemotions | E5-TUNE |  | 0.5579 | 0.3949 |
| goemotions | E5-FROZEN |  | 0.3305 | 0.2698 |
| movie_reviews_zh | E1 |  | 0.9952 | 0.8581 |
| movie_reviews_zh | E2 |  | 0.9935 | 0.8294 |
| movie_reviews_zh | E3-RNN |  | 0.9375 | 0.8281 |
| movie_reviews_zh | E3-LSTM |  | 0.9359 | 0.8348 |
| movie_reviews_zh | E3-BiLSTM |  | 0.9498 | 0.8346 |
| movie_reviews_zh | E5-TUNE |  | 0.9845 | 0.8349 |
| movie_reviews_zh | E5-FROZEN |  | 0.8831 | 0.7879 |

## Reproduction

Configure the two frozen dataset roots and the approved suite revision.
Run `batch-a smoke --output artifacts/batch-a-smoke`, followed by `batch-a run`, `batch-a audit`, `batch-a final-test`, and `batch-a report` with the same Batch A artifact location.

Per-label train/test support, test metrics, and confusion matrices: [class_metrics.json](class_metrics.json).

Independent audit evidence: `artifacts\batch-a\audit.json`.
