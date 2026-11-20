import type { TranslationKey } from '../i18n/translations';

interface ModelLabel {
  code: string;
  name: string;
  shortName?: string;
  detail: TranslationKey;
}

export const INFERENCE_MODEL_GROUPS: ReadonlyArray<{
  label: TranslationKey;
  models: readonly ModelLabel[];
}> = [
  {
    label: 'modelGroupClassic',
    models: [{ code: 'E1', name: 'SVM', detail: 'modelDetailSvm' }],
  },
  {
    label: 'modelGroupCnn',
    models: [
      { code: 'E2', name: 'TextCNN', shortName: 'CNN', detail: 'modelDetailCnnRandom' },
      { code: 'E5-TUNE', name: 'TextCNN-Tune', shortName: 'CNN-Tune', detail: 'modelDetailCnnTune' },
      {
        code: 'E5-FROZEN',
        name: 'TextCNN-Frozen',
        shortName: 'CNN-Frozen',
        detail: 'modelDetailCnnFrozen',
      },
    ],
  },
  {
    label: 'modelGroupRecurrent',
    models: [
      { code: 'E3-RNN', name: 'Vanilla RNN', shortName: 'RNN', detail: 'modelDetailRnn' },
      { code: 'E3-LSTM', name: 'LSTM', detail: 'modelDetailLstm' },
      { code: 'E3-BiLSTM', name: 'BiLSTM', detail: 'modelDetailBilstm' },
    ],
  },
  {
    label: 'modelGroupEncoder',
    models: [
      { code: 'B1', name: 'DistilBERT', detail: 'modelDetailDistilbert' },
      {
        code: 'B2',
        name: 'ModernBERT-base',
        shortName: 'ModernBERT-B',
        detail: 'modelDetailFullTune',
      },
      {
        code: 'B3',
        name: 'ModernBERT-large',
        shortName: 'ModernBERT-L',
        detail: 'modelDetailFullTune',
      },
      { code: 'B4', name: 'Chinese MacBERT-base', shortName: 'MacBERT', detail: 'modelDetailMacbert' },
    ],
  },
  {
    label: 'modelGroupLlm',
    models: [{ code: 'Qwen3.5-4B', name: 'Qwen3.5-4B', detail: 'modelDetailQwen' }],
  },
];

export const MODEL_LABELS_BY_CODE: ReadonlyMap<string, ModelLabel> = new Map(
  INFERENCE_MODEL_GROUPS.flatMap(({ models }) => models.map((model) => [model.code, model] as const)),
);

export function modelShortName(code: string): string {
  const label = MODEL_LABELS_BY_CODE.get(code);
  return label?.shortName ?? label?.name ?? code;
}
