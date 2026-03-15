"""
Pydantic schema for LegalRisk-LLM structured outputs.
Serves as the single source of truth for risk assessment data validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class RiskAssessment(BaseModel):
    """
    Structured model for legal clause risk assessment.

    This schema enforces data quality constraints for LLM-generated
    risk evaluations across the entire pipeline.
    """

    clause_type: str = Field(
        ...,
        description="The legal category of the clause"
    )

    risk_level: Literal["Low", "Medium", "High", "Critical"] = Field(
        ...,
        description="Categorical risk assessment"
    )

    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized risk score"
    )

    key_concerns: list[str] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="Top 3 risk factors"
    )

    recommendation: str = Field(
        ...,
        description="Actionable legal advice"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the assessment"
    )

    @field_validator('key_concerns')
    @classmethod
    def validate_concerns_non_empty(cls, v: list[str]) -> list[str]:
        """Ensure each concern is a non-empty string."""
        if any(not concern.strip() for concern in v):
            raise ValueError("All key concerns must be non-empty strings")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "clause_type": "Indemnification",
                    "risk_level": "High",
                    "risk_score": 0.78,
                    "key_concerns": [
                        "Unlimited liability exposure",
                        "Broad indemnification scope",
                        "No cap on damages"
                    ],
                    "recommendation": "Negotiate liability cap and exclude consequential damages",
                    "confidence": 0.92
                }
            ]
        }
    }
