from typing import List, Optional   
from pydantic import BaseModel

class ResumePerson(BaseModel):
    """Resume person model."""
    name: str
    email: Optional[str] = ""
    phone: Optional[str] = ""

class ResumeCompany(BaseModel):
    """Resume company model."""
    name: str
    role: Optional[str] = ""
    start_date: Optional[str] = ""
    end_date: Optional[str] = ""

class ResumeEducation(BaseModel):
    """Resume education model."""
    institution: str
    degree: Optional[str] = ""
    year: Optional[str] = ""

class ResumeEntities(BaseModel):
    """Resume entities model."""
    person: Optional[ResumePerson] = None
    companies: Optional[List[ResumeCompany]] = []
    skills: Optional[List[str]] = []
    education: Optional[List[ResumeEducation]] = []
    certifications: Optional[List[str]] = []
