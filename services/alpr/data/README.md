# data/

Local scratch space. **Everything here is gitignored.**

Datasets are built in Phase 1 and published to the Hugging Face Hub; Colab runtimes pull them
from there. Nothing in this directory is a source of truth — deleting it should cost nothing but
a re-download.

Two reasons the real data never lands in git:

1. Image datasets are gigabytes, and git handles them badly forever.
2. German plates are personal data under GDPR. Raw capture footage must be stripped and blurred
   (Phase 1) before it goes anywhere shareable, and git history cannot be un-shared.
