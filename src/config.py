"""
Preprocessing Configuration for URFD RGB Baseline
"""

import os

class PreprocessingConfig:
    # Target Spatial & Temporal Specifications
    TARGET_FPS = 25.0
    TARGET_WIDTH = 320
    TARGET_HEIGHT = 240
    WINDOW_SIZE = 50  # frames (2.0 seconds at 25 FPS)
    STRIDE = 25       # frames (1.0 second, 50% overlap)
    RESIZE_METHOD = "LANCZOS"
    
    # Deterministic Split Assignments (from R&D/split_strategy.md, seed=42)
    URFD_TRAIN_EVENTS = [
        'adl-01', 'adl-02', 'adl-05', 'adl-06', 'adl-08', 'adl-10', 'adl-11', 'adl-12', 'adl-13',
        'adl-17', 'adl-19', 'adl-20', 'adl-22', 'adl-23', 'adl-24', 'adl-26', 'adl-27', 'adl-29',
        'adl-30', 'adl-31', 'adl-32', 'adl-33', 'adl-34', 'adl-35', 'adl-36', 'adl-37', 'adl-38', 'adl-39',
        'fall-01', 'fall-02', 'fall-03', 'fall-04', 'fall-05', 'fall-06', 'fall-09', 'fall-10',
        'fall-12', 'fall-13', 'fall-14', 'fall-16', 'fall-17', 'fall-18', 'fall-22', 'fall-23',
        'fall-24', 'fall-25', 'fall-26', 'fall-28', 'fall-29'
    ]
    
    URFD_VAL_EVENTS = [
        'adl-04', 'adl-09', 'adl-14', 'adl-18', 'adl-25', 'adl-40',
        'fall-19', 'fall-21', 'fall-27', 'fall-30'
    ]
    
    URFD_TEST_EVENTS = [
        'adl-03', 'adl-07', 'adl-15', 'adl-16', 'adl-21', 'adl-28',
        'fall-07', 'fall-08', 'fall-11', 'fall-15', 'fall-20'
    ]

    @classmethod
    def get_event_partition(cls, event_id):
        if event_id in cls.URFD_TRAIN_EVENTS:
            return "train"
        elif event_id in cls.URFD_VAL_EVENTS:
            return "val"
        elif event_id in cls.URFD_TEST_EVENTS:
            return "test"
        else:
            raise ValueError(f"Unknown event_id {event_id} not in configured splits!")
