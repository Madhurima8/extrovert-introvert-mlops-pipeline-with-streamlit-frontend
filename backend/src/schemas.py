from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal

# pydantic model to validate input features for prediction
class PersonalityFeatures(BaseModel):
    Time_spent_Alone: Annotated[float, Field(ge=0, le=11, description="Hours spent alone per day")]
    # Stage_fear: Annotated[str, Field(description="Fear of public speaking (Yes/No)")]
    Stage_fear: Literal["Yes", "No"]
    Social_event_attendance: Annotated[float, Field(ge=0, le=10, description="Frequency of attending social events")]
    Going_outside: Annotated[float, Field(ge=0, le=7, description="Frequency of going outside")]
    # Drained_after_socializing: Annotated[str, Field(description="Feeling drained after social interactions (Yes/No)")]
    Drained_after_socializing: Literal["Yes", "No"]
    Friends_circle_size: Annotated[float, Field(ge=0, le=15, description="Size of friends circle")]
    Post_frequency: Annotated[float, Field(ge=0, le=10, description="Frequency of posting on social media")]

    @field_validator("Stage_fear", "Drained_after_socializing", mode="before")
    @classmethod
    def normalize_yes_no(cls, value):
        if isinstance(value, str):
            value = value.strip().lower()

            if value == "yes":
                return "Yes"

            if value == "no":
                return "No"

        return value