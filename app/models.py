from sqlalchemy import Column, Integer, String, ForeignKey, JSON, Date, Float, Boolean, Text
from sqlalchemy.orm import relationship
from app.db import Base


class Aircraft(Base):
    __tablename__ = "aircraft"

    id              = Column(Integer, primary_key=True, index=True)
    registration    = Column(String, unique=True, index=True, nullable=False)
    msn             = Column(String, nullable=False)
    aircraft_type   = Column(String, nullable=False)
    operator_name   = Column(String, nullable=False)
    manufacturer    = Column(String, nullable=True)
    manufacture_date= Column(String, nullable=True)   # store as ISO string yyyy-mm-dd
    total_fh        = Column(Float,  default=0.0)     # total flight hours
    total_fc        = Column(Integer, default=0)       # total flight cycles
    status          = Column(String, default="active") # active | grounded | aog | stored

    documents        = relationship("Document",        back_populates="aircraft", cascade="all, delete-orphan")
    ads              = relationship("AirworthinessDirective", back_populates="aircraft", cascade="all, delete-orphan")
    service_bulletins= relationship("ServiceBulletin", back_populates="aircraft", cascade="all, delete-orphan")
    maintenance_checks= relationship("MaintenanceCheck", back_populates="aircraft", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id              = Column(Integer, primary_key=True, index=True)
    aircraft_id     = Column(Integer, ForeignKey("aircraft.id"), nullable=False)
    file_name       = Column(String, nullable=False)
    file_path       = Column(String, nullable=False)
    document_type   = Column(String, nullable=True)   # CRS | Work Order | AD Compliance | SB Compliance | Manual | Other
    chapter         = Column(String, nullable=True)    # ATA chapter string (auto-detected)
    extracted_text  = Column(Text,   nullable=True)
    structured_data = Column(JSON,   nullable=True)
    uploaded_at     = Column(String, nullable=True)    # ISO datetime string

    aircraft = relationship("Aircraft", back_populates="documents")


class AirworthinessDirective(Base):
    """Tracks AD compliance per aircraft."""
    __tablename__ = "airworthiness_directives"

    id              = Column(Integer, primary_key=True, index=True)
    aircraft_id     = Column(Integer, ForeignKey("aircraft.id"), nullable=False)
    ad_number       = Column(String, nullable=False, index=True)   # e.g. AD 2024-15-12
    ata_chapter     = Column(String, nullable=True)
    description     = Column(Text,   nullable=True)
    effective_date  = Column(String, nullable=True)    # yyyy-mm-dd
    due_date        = Column(String, nullable=True)    # yyyy-mm-dd
    compliance_fh   = Column(Float,  nullable=True)    # flight hours threshold
    compliance_fc   = Column(Integer, nullable=True)   # flight cycles threshold
    status          = Column(String, default="open")   # open | compliant | overdue | n/a
    compliance_doc_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    notes           = Column(Text, nullable=True)

    aircraft         = relationship("Aircraft",  back_populates="ads")
    compliance_doc   = relationship("Document",  foreign_keys=[compliance_doc_id])


class ServiceBulletin(Base):
    """Tracks SB compliance per aircraft."""
    __tablename__ = "service_bulletins"

    id              = Column(Integer, primary_key=True, index=True)
    aircraft_id     = Column(Integer, ForeignKey("aircraft.id"), nullable=False)
    sb_number       = Column(String, nullable=False, index=True)
    ata_chapter     = Column(String, nullable=True)
    description     = Column(Text,   nullable=True)
    revision        = Column(String, nullable=True)
    issue_date      = Column(String, nullable=True)
    due_date        = Column(String, nullable=True)
    compliance_fh   = Column(Float,  nullable=True)
    status          = Column(String, default="open")   # open | compliant | not-applicable
    compliance_doc_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    notes           = Column(Text, nullable=True)

    aircraft       = relationship("Aircraft", back_populates="service_bulletins")
    compliance_doc = relationship("Document", foreign_keys=[compliance_doc_id])


class MaintenanceCheck(Base):
    """A-check, B-check, C-check, D-check scheduling."""
    __tablename__ = "maintenance_checks"

    id              = Column(Integer, primary_key=True, index=True)
    aircraft_id     = Column(Integer, ForeignKey("aircraft.id"), nullable=False)
    check_type      = Column(String, nullable=False)   # A | B | C | D
    last_completed  = Column(String, nullable=True)    # yyyy-mm-dd
    next_due_date   = Column(String, nullable=True)    # yyyy-mm-dd
    next_due_fh     = Column(Float,  nullable=True)    # next due flight hours
    next_due_fc     = Column(Integer, nullable=True)   # next due flight cycles
    status          = Column(String, default="on-track")  # on-track | due-soon | overdue
    work_order      = Column(String, nullable=True)
    notes           = Column(Text, nullable=True)

    aircraft = relationship("Aircraft", back_populates="maintenance_checks")
