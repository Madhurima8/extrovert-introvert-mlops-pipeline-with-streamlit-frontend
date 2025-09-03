from pydantic import BaseModel, Field

class PersonalityFeatures(BaseModel):
    Time_spent_Alone: str
    Stage_fear: str
    Social_event_attendance: str
    Going_outside: str
    Drained_after_socializing: str
    Friends_circle_size: str
    Post_frequency: str