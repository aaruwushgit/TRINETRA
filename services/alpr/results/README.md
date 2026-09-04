# Training artifacts

The Phase 2 run that produced the detector reported in the top-level
[README](../README.md). Kept because a Colab runtime is ephemeral: the run
directory died with the session, and these files are what remain of it.

| File | Contents |
|---|---|
| `results.png` | Loss curves and metrics across all 100 epochs |
| `results.csv` | The same, per epoch, as numbers |
| `confusion_matrix.png` | Single-class confusion matrix on the validation split |
| `args.yaml` | Every argument Ultralytics ran with |

`args.yaml` is the provenance record. Ultralytics changes augmentation defaults
between minor releases, so the config alone does not pin a run — reproducing
this result needs the arguments *and* the library version that consumed them.

**The weights are not here.** `best.pt` is 22.5 MB, and binaries do not belong
in git history where they can never be removed. They are published on the Hub:
[Babblu2821/alpr-plate-detector](https://huggingface.co/Babblu2821/alpr-plate-detector).

`MODEL_CARD.md` is the source of that repo's README — edit it here and re-upload,
so the two cannot drift.

## The run

| | |
|---|---|
| Model | YOLOv8s |
| Epochs | 100 (no early stop; patience 25 never triggered) |
| Hardware | Colab T4, 1.25 h |
| Final val | mAP@50 **0.9911**, mAP@50-95 **0.8296** |
| Test | mAP@50 **0.9921**, recall **0.9917** |

Test scoring slightly above validation is expected here: validation drove model
selection, and with 465 images the gap sits inside sampling noise. The
[leakage audit](../README.md#the-test-set-was-audited-for-leakage) checked the
other explanation and found the contamination was not carrying the score.
