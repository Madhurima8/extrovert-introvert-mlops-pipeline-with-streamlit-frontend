from pydantic import BaseModel, Field
from typing import Annotated

# pydantic model to validate input features for prediction
class PersonalityFeatures(BaseModel):
    Time_spent_Alone: Annotated[float, Field(ge=0, le=11, description="Hours spent alone per day")]
    Stage_fear: Annotated[str, Field(description="Fear of public speaking (Yes/No)")]
    Social_event_attendance: Annotated[float, Field(ge=0, le=10, description="Frequency of attending social events")]
    Going_outside: Annotated[float, Field(ge=0, le=7, description="Frequency of going outside")]
    Drained_after_socializing: Annotated[str, Field(description="Feeling drained after social interactions (Yes/No)")]
    Friends_circle_size: Annotated[float, Field(ge=0, le=15, description="Size of friends circle")]
    Post_frequency: Annotated[float, Field(ge=0, le=10, description="Frequency of posting on social media")]