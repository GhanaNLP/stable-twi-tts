"""Twi and Ghanaian English speech synthesis."""

from .g2p import PhonemeError, mixed_phonemes, phonemize, twi_phonemes
from .tts import GhanaTTS, Synthesis
from .voices import Voice, VoiceRegistry

__version__ = "0.1.0"
__all__ = ["GhanaTTS", "Synthesis", "Voice", "VoiceRegistry", "PhonemeError",
           "phonemize", "twi_phonemes", "mixed_phonemes"]
