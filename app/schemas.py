from pydantic import BaseModel
from typing import Optional, List


# ─── Aircraft ────────────────────────────────────────────────────────────────

class AircraftCreate(BaseModel):
    registration:     str
    msn:              str
    aircraft_type:    str
    operator_name:    str
    manufacturer:     Optional[str] = None
    manufacture_date: Optional[str] = None   # yyyy-mm-dd
    total_fh:         Optional[float] = 0.0
    total_fc:         Optional[int]   = 0
    status:           Optional[str]   = "active"


class AircraftResponse(BaseModel):
    id:               int
    registration:     str
    msn:              str
    aircraft_type:    str
    operator_name:    str
    manufacturer:     Optional[str]
    manufacture_date: Optional[str]
    total_fh:         Optional[float]
    total_fc:         Optional[int]
    status:           str

    class Config:
        from_attributes = True


# ─── Document ─────────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id:             int
    aircraft_id:    int
    file_name:      str
    file_path:      str
    document_type:  Optional[str]
    chapter:        Optional[str]
    extracted_text: Optional[str]
    structured_data: Optional[dict]
    uploaded_at:    Optional[str]

    class Config:
        from_attributes = True


# ─── Airworthiness Directive ──────────────────────────────────────────────────

class ADCreate(BaseModel):
    aircraft_id:       int
    ad_number:         str
    ata_chapter:       Optional[str] = None
    description:       Optional[str] = None
    effective_date:    Optional[str] = None
    due_date:          Optional[str] = None
    compliance_fh:     Optional[float] = None
    compliance_fc:     Optional[int]   = None
    status:            Optional[str]   = "open"
    compliance_doc_id: Optional[int]   = None
    notes:             Optional[str]   = None


class ADResponse(BaseModel):
    id:                int
    aircraft_id:       int
    ad_number:         str
    ata_chapter:       Optional[str]
    description:       Optional[str]
    effective_date:    Optional[str]
    due_date:          Optional[str]
    compliance_fh:     Optional[float]
    compliance_fc:     Optional[int]
    status:            str
    compliance_doc_id: Optional[int]
    notes:             Optional[str]

    class Config:
        from_attributes = True


# ─── Service Bulletin ─────────────────────────────────────────────────────────

class SBCreate(BaseModel):
    aircraft_id:       int
    sb_number:         str
    ata_chapter:       Optional[str]  = None
    description:       Optional[str]  = None
    revision:          Optional[str]  = None
    issue_date:        Optional[str]  = None
    due_date:          Optional[str]  = None
    compliance_fh:     Optional[float]= None
    status:            Optional[str]  = "open"
    compliance_doc_id: Optional[int]  = None
    notes:             Optional[str]  = None


class SBResponse(BaseModel):
    id:                int
    aircraft_id:       int
    sb_number:         str
    ata_chapter:       Optional[str]
    description:       Optional[str]
    revision:          Optional[str]
    issue_date:        Optional[str]
    due_date:          Optional[str]
    compliance_fh:     Optional[float]
    status:            str
    compliance_doc_id: Optional[int]
    notes:             Optional[str]

    class Config:
        from_attributes = True


# ─── Maintenance Check ────────────────────────────────────────────────────────

class CheckCreate(BaseModel):
    aircraft_id:    int
    check_type:     str           # A | B | C | D
    last_completed: Optional[str] = None
    next_due_date:  Optional[str] = None
    next_due_fh:    Optional[float] = None
    next_due_fc:    Optional[int]   = None
    status:         Optional[str]   = "on-track"
    work_order:     Optional[str]   = None
    notes:          Optional[str]   = None


class CheckResponse(BaseModel):
    id:             int
    aircraft_id:    int
    check_type:     str
    last_completed: Optional[str]
    next_due_date:  Optional[str]
    next_due_fh:    Optional[float]
    next_due_fc:    Optional[int]
    status:         str
    work_order:     Optional[str]
    notes:          Optional[str]

    class Config:
        from_attributes = True


# ─── Stats ────────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    total_aircraft:    int
    total_documents:   int
    overdue_ads:       int
    due_soon_ads:      int
    compliant_ads:     int
    overdue_checks:    int
    due_soon_checks:   int
    ai_extracted_docs: int
    ata_distribution:  dict
    status_distribution: dict
    doc_type_distribution: dict

# ─── Authentication ─────────────────────────────────────────────

class UserSignup(BaseModel):
    name: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    name: str
    email: str

    class Config:
        from_attributes = True