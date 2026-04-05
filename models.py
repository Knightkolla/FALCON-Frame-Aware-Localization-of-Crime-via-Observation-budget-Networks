"""
FALCON OpenEnv Models
This root-level models.py acts as the namespace for the OpenEnv package structure validator.
It re-exports the identical OpenEnv Pydantic definitions from our actual environment architecture.
"""

from server.models import (
    LowResFeatures, 
    ExpandedFeatures, 
    SegmentObservation, 
    Observation, 
    Action, 
    Reward, 
    StepResponse, 
    EpisodeState
)

# OpenEnv structure validated
