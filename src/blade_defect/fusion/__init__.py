"""RGB-T registration and fusion extension points."""

from .base import FusionResult, ImageFusion, ImageRegistration
from .identity import IdentityFusion, IdentityRegistration

__all__ = ["FusionResult", "ImageFusion", "ImageRegistration", "IdentityFusion", "IdentityRegistration"]
