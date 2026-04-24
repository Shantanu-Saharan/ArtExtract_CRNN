# Top-25 Results Summary

This file keeps the small set of result artifacts that explain the submission without dragging along the full experiment history.

## Selected runs

| Method | Source | Notes | Mean F1 |
| --- | --- | --- | --- |
| Stacked meta-ensemble | `logs/Z3_106113.out` | Final submission result | `0.8646` |
| SigLIP 2 fine-tune | `logs/siglip2_ft_105918.out` | Strongest single-model branch; re-evaluates to `0.8596` in the final ensemble report | `0.8515` in training log |
| DINOv2 + ArcFace (A40) | `logs/top25_a40_best_100772.out` | Best DINO-based training run kept for comparison | `0.8177` |

## Final ensemble metrics

Source: `logs/Z3_106113.out` and `reports/top25_meta_ensemble_report.json`

| Task | Macro F1 | Accuracy |
| --- | --- | --- |
| Artist | `0.9573` | `0.9608` |
| Style | `0.7741` | `0.7881` |
| Genre | `0.8624` | `0.8486` |
| Mean | `0.8646` | - |

## Files kept on purpose

- `logs/Z3_106113.out`
- `logs/siglip2_ft_105918.out`
- `logs/top25_a40_best_100772.out`
- `reports/top25_meta_ensemble_report.json`
- `reports/top25_meta_models.pkl`

## Short read on the progression

- The strongest single checkpoint in the final stack is `siglip2_ft`, which scores `0.8596` mean F1 when re-evaluated in the ensemble run.
- The DINO branch remains useful even though it trails SigLIP 2 as a standalone model; it adds complementary signal for artist and genre.
- The stacked meta-ensemble gives a small but real lift over the best single model: `+0.0050` mean F1.
