export type DatasetId = 'goemotions' | 'movie_reviews_zh';
export type Locale = 'zh' | 'en';
export type ChapterId = 'inference' | 'capabilities' | 'data';
export type ScoreKind = 'decision' | 'sigmoid' | 'softmax_positive';

export interface SplitStats {
  raw_count: number;
  frozen_count: number;
  excluded_count: number;
  ratio: number;
  sha256: string;
}

export interface LabelSupportRow {
  id: number;
  source_id?: number;
  name: string;
  raw_train: number;
  v1_train: number;
  raw_val: number;
  v1_val: number;
  test: number;
  train_rate?: number;
}

export interface TokenizationProfile {
  group: string;
  max_length: number;
  train_p50: number;
  train_p95: number;
  train_p99: number;
  train_truncated: number;
  val_truncated: number;
  test_truncated?: number;
}

export interface DatasetOverview {
  dataset: DatasetId;
  task_type: 'multilabel' | 'binary';
  label_names: string[];
  label_ids: number[];
  positive_raw_id: number | null;
  manifest_sha256: string;
  splits: {
    train: SplitStats;
    validation: SplitStats;
    test: SplitStats;
  };
  total_raw: number;
  total_frozen: number;
  total_label_annotations?: {
    train: number;
    validation: number;
    test: number;
  };
  avg_labels_per_sample?: {
    train: number;
    validation: number;
    test: number;
  };
  audit_summary?: {
    status: string;
    zero_cross_split_overlap: boolean;
    excluded_total: number;
    blank_rows_skipped_total: number;
  };
  exclusion_reasons: Record<string, number>;
  blank_rows_skipped: Record<string, number>;
  cross_split_overlap: Record<string, { shared_texts: number; texts_with_conflicting_label_sets: number }>;
  label_counts: LabelSupportRow[];
  tokenization_profiles: TokenizationProfile[];
}

export interface InferenceBenchmark {
  ms_per_sample: number;
  samples_per_second: number;
  batch_size: number;
  device: string;
  source: string;
}

export interface MeanSd {
  n: number;
  mean: number;
  sd: number;
}

export interface ThreeSeedSummary {
  val_macro_f1: MeanSd;
  test_macro_f1: MeanSd;
  test_micro_f1: MeanSd;
  secondary_metric: MeanSd;
  by_seed: Record<string, number>;
  val_by_seed: Record<string, number>;
  roc_auc?: { by_seed: Record<string, number>; mean: number; sample_sd: number };
  pr_auc?: { by_seed: Record<string, number>; mean: number; sample_sd: number };
}

export interface LeaderboardEntry {
  model_id: string;
  model_code: string;
  display_name: string;
  short_label: string;
  batch: string;
  family: string;
  score_kind: ScoreKind;
  loss_function?: string;
  param_count?: number;
  trainable_params?: number;
  param_label?: string;
  inference_benchmark?: InferenceBenchmark;
  threshold: number;
  best_epoch: number | null;
  stop_reason: string;
  run_id: string;
  seed42: {
    val_macro_f1: number;
    test_macro_f1: number;
    test_micro_f1: number;
    secondary_metric: number;
    samples_f1?: number | null;
    hamming_loss?: number | null;
    positive_f1?: number | null;
    positive_precision?: number | null;
    positive_recall?: number | null;
    roc_auc?: number | null;
    pr_auc?: number | null;
  };
  three_seed: ThreeSeedSummary | null;
  parameters: Record<string, unknown>;
}

export interface FormalRun {
  id: string;
  dataset: DatasetId;
  batch: string;
  family: string;
  model: string;
  display_name: string;
  short_label: string;
  role: string;
  seed: number;
  run_id: string;
  score_kind: ScoreKind;
  threshold: number;
  best_epoch: number | null;
  val_macro_f1: number;
  test_macro_f1: number;
  test_micro_f1: number;
  secondary_metric: number;
  secondary_metric_name: string;
  samples_f1?: number | null;
  hamming_loss?: number | null;
  positive_f1?: number | null;
  roc_auc?: number | null;
  pr_auc?: number | null;
  stop_reason?: string | null;
  stage?: string;
}

export interface BootstrapInterval {
  difference: number;
  confidence_interval: [number, number];
}

export interface ModernBertPairedComparison {
  by_seed: Record<string, Record<string, BootstrapInterval>>;
  mean_of_fixed_seeds: Record<string, BootstrapInterval>;
  protocol: {
    confidence: number;
    metrics: string[];
    models: [string, string];
    n_resamples: number;
  };
  test_rows: number;
}

export interface ZhBootstrapComparison {
  model_a: string;
  model_b: string;
  comparison: string;
  difference: number;
  ci_low: number;
  ci_high: number;
  n: number;
}

export type BootstrapGroup = 'architecture' | 'ablation';

export interface AdjustedBootstrapComparison {
  model_a: string;
  model_b: string;
  difference: number;
  ci_low: number;
  ci_high: number;
  originally_planned: boolean;
}

export interface AdjustedBootstrapBundle {
  dataset: DatasetId;
  seed: number;
  n: number;
  correction: 'bonferroni';
  familywise_alpha: number;
  comparison_count: number;
  confidence_level: number;
  n_resamples: number;
  resampling_seed: number;
  groups: Record<BootstrapGroup, AdjustedBootstrapComparison[]>;
}

export interface PerLabelModelMetrics {
  f1: number;
  seed42_f1: number;
  precision: number;
  recall: number;
  by_seed?: Record<string, number>;
}

export interface PerLabelGoEmotionsRow {
  id: number;
  name: string;
  train_support: number;
  val_support: number;
  test_support: number;
  models: Record<string, PerLabelModelMetrics>;
}

export interface EmbeddingAblationRow {
  model_code: string;
  display_name: string;
  embedding_type: string;
  source?: string;
  vocabulary_coverage?: number;
  token_coverage?: number;
  frozen_verified: boolean;
  seed42_test_macro_f1: number;
  three_seed_mean: number;
  three_seed_sd: number;
  by_seed: Record<string, number>;
}

export interface ConfusionMatrixRow {
  id: string;
  model_code: string;
  display_name: string;
  seed: number;
  threshold: number;
  score_kind: ScoreKind;
  matrix: [[number, number], [number, number]];
  accuracy: number;
  macro_f1: number;
  positive_precision: number;
  positive_recall: number;
  positive_f1: number;
  roc_auc: number;
  pr_auc: number;
}

export interface EpochRecord {
  epoch: number;
  val_macro_f1: number;
  threshold: number;
  loss?: number | null;
  grad_norm?: number;
  learning_rate?: number;
  best_epoch?: number;
  global_step?: number;
  elapsed_seconds?: number;
  validation_seconds?: number;
  stale_epochs?: number;
}

export interface HpoTrial {
  trial_number: number;
  label: string;
  status: 'PASS' | 'PRUNED' | string;
  macro_f1: number | null;
  best_epoch: number | null;
  elapsed_seconds: number;
  reason: string;
  run_id: string;
  is_winner: boolean;
  params: Record<string, unknown>;
}

export interface HpoStudy {
  study: string;
  dataset: DatasetId;
  model: string;
  display_name: string;
  trials: HpoTrial[];
}

export interface BatchBS1LrSlot {
  slot: string;
  kind: 'baseline' | 'alternate';
  model: string;
  display_name: string;
  dataset: DatasetId;
  learning_rate: number;
  val_macro_f1: number;
  threshold: number;
  best_epoch: number;
  epochs_completed: number;
  stop_reason: string;
  elapsed_seconds: number;
  peak_allocated_gib: number;
  selected: boolean;
  run_id: string;
}

export interface SamplePreset {
  sample_id: string;
  title_en: string;
  title_zh: string;
  text: string;
  segmented_text?: string;
  gold_labels: string[];
}

export interface TrainingRunCurve {
  id: string;
  dataset: DatasetId;
  model: string;
  display_name: string;
  short_label: string;
  batch: string;
  seed: number;
  role: string;
  run_id: string;
  loss_function: string;
  score_kind: ScoreKind;
  threshold: number;
  best_epoch: number | null;
  epochs_completed: number;
  stop_reason?: string | null;
  val_macro_f1: number;
  test_macro_f1: number;
  elapsed_seconds?: number | null;
  global_step?: number | null;
  checkpoint_path: string;
  checkpoint_sha256?: string | null;
  epochs: EpochRecord[];
}

export interface DashboardBundle {
  meta: {
    generated_at_utc: string;
    freezes: Record<string, string>;
    totals: {
      formal_runs: number;
      goemotions_runs: number;
      movie_reviews_zh_runs: number;
      frozen_samples: number;
      test_predictions_total: number;
      models_goemotions: number;
      models_movie_reviews_zh: number;
    };
  };
  datasets: Record<DatasetId, DatasetOverview>;
  leaderboard: Record<DatasetId, LeaderboardEntry[]>;
  formal_runs: FormalRun[];
  paired_comparisons: {
    goemotions_b3_vs_b2: ModernBertPairedComparison;
    movie_reviews_zh_batch_a: ZhBootstrapComparison[];
    movie_reviews_zh_families: AdjustedBootstrapBundle;
  };
  per_label_goemotions: PerLabelGoEmotionsRow[];
  ablation_embedding: Record<DatasetId, EmbeddingAblationRow[]>;
  confusion_matrices_zh: ConfusionMatrixRow[];
  training_curves: {
    runs?: TrainingRunCurve[];
    batch_a: Record<DatasetId, Record<string, EpochRecord[]>>;
    batch_b_encoders: Record<
      string,
      {
        model: string;
        dataset: DatasetId;
        seed: number;
        best_epoch: number;
        threshold: number;
        epochs: EpochRecord[];
      }
    >;
    batch_b_qwen: Record<
      string,
      {
        seed: number;
        best_epoch: number;
        threshold: number;
        epochs: EpochRecord[];
      }
    >;
  };
  hpo_trials: {
    batch_a_studies: Record<string, HpoStudy>;
    batch_b_s1_lr: BatchBS1LrSlot[];
    batch_b_b1_budget: {
      gain: number;
      minimum_macro_f1_gain: number;
      s1_macro_f1: number;
      s2_macro_f1: number;
      selected_budget: string;
    };
  };
  quality_flags_a: Array<{
    dataset: DatasetId;
    experiment: string;
    flags: string[];
    train_macro_f1: number;
    validation_macro_f1: number;
  }>;
  sample_presets: Record<DatasetId, SamplePreset[]>;
}

export interface ApiModelInfo {
  model_id: string;
  model_code: string;
  display_name: string;
  batch: string;
  family: string;
  dataset: DatasetId;
  default_seed: number;
  available_seeds: number[];
  run_id: string;
  score_kind: ScoreKind;
  threshold: number;
  labels: string[];
  validation_macro_f1: number;
  test_macro_f1: number | null;
  max_length: number;
  runtime: 'local' | 'wsl2_worker';
  checkpoint_exists: boolean;
  available: boolean;
  loaded: boolean;
}

export interface ApiHealth {
  status: string;
  mode: string;
  gpu: {
    cuda_available: boolean;
    device_name?: string;
    vram_free_mib?: number;
    vram_total_mib?: number;
  };
  loaded_models: string[];
  qwen_worker: {
    configured_url: string;
    online: boolean;
    status?: string;
    loaded_seed?: number | null;
    loaded_seeds?: number[];
    device?: string | null;
    vram_free_mib?: number | null;
    detail?: string;
    start_command?: string;
  };
}

export interface TopKPrediction {
  label: string;
  score: number;
  threshold: number;
  delta: number;
  active: boolean;
}

export interface PredictionResult {
  model_id: string;
  model_code: string;
  display_name: string;
  family: string;
  dataset: DatasetId;
  seed: number;
  run_id: string;
  score_kind: ScoreKind;
  threshold: number;
  labels: string[];
  scores: number[];
  positive_score: number | null;
  predicted_labels: string[];
  top_k: TopKPrediction[];
  latency_ms: number;
  device: string;
  preprocessing: {
    segmentation_applied: boolean;
    token_count: number;
    truncated: boolean;
    preview_tokens: string[];
  };
}
