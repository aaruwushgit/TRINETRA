"""ALPR — automatic license plate recognition with Excel logging.

The package is layered so each roadmap phase adds one module without
disturbing the ones before it:

    alpr.sources    frame sources (video file, webcam, RTSP)     Phase 1 / 9
    alpr.data       dataset build, split, stats                  Phase 1
    alpr.train      YOLO detector training (Colab T4)            Phase 2
    alpr.detect     detector inference + evaluation              Phase 3
    alpr.ocr        PaddleOCR over plate crops                   Phase 4
    alpr.plates     country-aware validation (India, Germany)    Phase 5
    alpr.track      tracking + multi-frame voting                Phase 6
    alpr.excel      append-safe Excel logging                    Phase 7
    alpr.pipeline   end-to-end orchestration                     Phase 7
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
