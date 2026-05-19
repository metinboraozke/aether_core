"""Modül 4 — Hiyerarşik Cascade DRI + Voxel Tahmini (The Brain).

Bkz: README.md → Modül 4, upcoming/03_MODUL_4_BRAIN.txt
"""

from __future__ import annotations

from .detection_head import DetectionHead, detection_loss
from .tsdf_decoder import BayesianTSDFDecoder, tsdf_loss
from .recognition_head import RecognitionHead, recognition_loss
from .identification_head import (
    IdentificationHead, identification_loss, cosine_lookup,
)
from .tracker import (
    DopplerRingBuffer, SlidingTracker, TrackingEngine, tracker_loss,
)


__all__ = [
    # Detection
    "DetectionHead", "detection_loss",
    # TSDF
    "BayesianTSDFDecoder", "tsdf_loss",
    # Recognition
    "RecognitionHead", "recognition_loss",
    # Identification
    "IdentificationHead", "identification_loss", "cosine_lookup",
    # Tracker
    "DopplerRingBuffer", "SlidingTracker", "TrackingEngine", "tracker_loss",
]
