from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    question: str
    session_id: str | None = None
