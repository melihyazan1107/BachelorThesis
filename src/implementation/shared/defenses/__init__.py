"""Zentrale Schnittstelle für alle Verteidigungsmechanismen

Exportierte API:
- NullDefense: Eine Verteidigung, die nichts tut
- DEFENSE_CHOICES: Ein Tupel aller gültigen und Verteidigungs-Namen.
- DEFENSE_GROUPS: Ein Mapping, das Verteidigungen in logische Kategorien einteilt 
- build_defense: Eine Funktion, die dynamisch die richtige Defense-Klasse baut
- validate_defense: Prüft, ob ein übergebener Name registriert ist
"""
from implementation.shared.defenses.base import NullDefense
from implementation.shared.defenses.registry import (
    DEFENSE_CHOICES,
    DEFENSE_GROUPS,
    build_defense,
    validate_defense,
)

__all__ = [
    "NullDefense",
    "DEFENSE_CHOICES",
    "DEFENSE_GROUPS",
    "build_defense",
    "validate_defense",
]
