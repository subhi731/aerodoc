"""
AeroDoc — Aircraft Technical Records Platform
FastAPI backend  v6.0
Extraction pipeline (priority order):
  1. Claude API  — best, handles ANY airline format (needs ANTHROPIC_API_KEY)
  2. Ollama text — local llama3.2 for typed PDFs (free, no internet)
  3. LLaVA vision — local vision model for handwritten/scanned PDFs (free)
  4. Regex        — always runs as supplement / final fallback
Set EXTRACTION_MODE in .env: claude | ollama | regex
"""

from pathlib import Path
import shutil, re, json, os, base64
from datetime import date, datetime

from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import pdfplumber
import httpx

from app.db import Base, engine, get_db
from app.models import (
    Aircraft,
    Document,
    AirworthinessDirective,
    ServiceBulletin,
    MaintenanceCheck,
    User
)
from app.schemas import (
    AircraftCreate, AircraftResponse,
    DocumentResponse,
    ADCreate, ADResponse,
    SBCreate, SBResponse,
    CheckCreate, CheckResponse,
    UserSignup,
    UserLogin,
    UserResponse
)
# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
EXTRACTION_MODE      = os.getenv("EXTRACTION_MODE",      "claude").lower()
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY",    "")
ANTHROPIC_API_URL    = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL         = "claude-sonnet-4-5"
OLLAMA_URL           = os.getenv("OLLAMA_URL",           "http://127.0.0.1:11434")
OLLAMA_URL_FALLBACK  = "http://localhost:11434"
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL",         "llama3.2")
OLLAMA_VISION_MODEL  = os.getenv("OLLAMA_VISION_MODEL",  "llava")
POPPLER_PATH         = os.getenv("POPPLER_PATH",         r"E:\poppler\poppler-24.08.0\Library\bin")

# ─────────────────────────────────────────────────────────────────────────────
# ATA MAP
# ─────────────────────────────────────────────────────────────────────────────
ATA_MAP = {
    "00":"General","05":"Time Limits / Maintenance Checks","06":"Dimensions and Areas",
    "07":"Lifting and Shoring","08":"Leveling and Weighing","09":"Towing and Taxiing",
    "10":"Parking, Mooring, Storage","11":"Placards and Markings","12":"Servicing",
    "20":"Standard Practices","21":"Air Conditioning","22":"Auto Flight",
    "23":"Communications","24":"Electrical Power","25":"Equipment / Furnishings",
    "26":"Fire Protection","27":"Flight Controls","28":"Fuel","29":"Hydraulic Power",
    "30":"Ice and Rain Protection","31":"Indicating / Recording Systems","32":"Landing Gear",
    "33":"Lights","34":"Navigation","35":"Oxygen","36":"Pneumatic","38":"Water / Waste",
    "49":"Auxiliary Power Unit","51":"Structures","52":"Doors","53":"Fuselage",
    "54":"Nacelles / Pylons","55":"Stabilizers","56":"Windows","57":"Wings",
    "71":"Power Plant","72":"Engine","73":"Engine Fuel and Control","74":"Ignition",
    "75":"Air","76":"Engine Controls","77":"Engine Indicating","78":"Exhaust",
    "79":"Oil","80":"Starting",
}

# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AeroDoc — Aircraft Technical Records Platform",
    version="6.0.0",
    description="Universal aviation records platform — Claude API + LLaVA vision + Regex. Any airline format.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

STORAGE_ROOT = Path("storage")
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

# ─── Authentication Helpers ─────────────────────────────────────

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(
    plain_password: str,
    hashed_password: str
):
    return pwd_context.verify(
        plain_password,
        hashed_password
    )

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — PDF TEXT EXTRACTION
# Multi-strategy extraction handles distorted, overlapping and garbled text
# ═════════════════════════════════════════════════════════════════════════════
def extract_text_from_pdf(file_path: Path) -> str:
    """
    Try multiple pdfplumber extraction strategies and combine results.
    Strategy 1: default extract_text()
    Strategy 2: relaxed tolerances for overlapping text
    Strategy 3: word-by-word reconstruction sorted by position (best for distorted PDFs)
    """
    results = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # Strategy 1 - default
                try:
                    t1 = page.extract_text() or ""
                    if t1.strip():
                        results.append(("default", t1))
                except: pass

                # Strategy 2 - relaxed tolerances
                try:
                    t2 = page.extract_text(x_tolerance=5, y_tolerance=5) or ""
                    if t2.strip():
                        results.append(("relaxed", t2))
                except: pass

                # Strategy 3 - word-by-word reconstruction by position
                try:
                    words = page.extract_words(
                        x_tolerance=4, y_tolerance=6,
                        keep_blank_chars=False, use_text_flow=True
                    )
                    if words:
                        words_sorted = sorted(words, key=lambda w: (round(w["top"]/8)*8, w["x0"]))
                        lines = {}
                        for w in words_sorted:
                            k = round(w["top"]/8)*8
                            lines.setdefault(k, []).append(w["text"])
                        t3 = "\n".join(" ".join(v) for v in lines.values())
                        if t3.strip():
                            results.append(("words", t3))
                except: pass

    except Exception as e:
        print(f"[PDF] Extraction error: {e}")
        return ""

    if not results:
        return ""

    def score(text):
        words = text.split()
        return sum(1 for w in words if len(w) > 2 and sum(c.isalpha() for c in w)/max(len(w),1) > 0.6)

    best = max(results, key=lambda r: score(r[1]))
    print(f"[PDF] Best strategy: {best[0]} ({score(best[1])} clean words)")

    # Combine all strategies for maximum regex coverage
    seen = {best[0]}
    all_text = best[1]
    for name, txt in results:
        if name not in seen:
            all_text += "\n\n" + txt
            seen.add(name)

    return all_text.strip()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — PDF TO IMAGES (for handwritten/scanned PDFs)
# ═════════════════════════════════════════════════════════════════════════════
def pdf_to_images_base64(file_path: Path, max_pages: int = 4) -> list[str]:
    """
    Convert PDF pages to base64-encoded JPEG images.
    Returns list of base64 strings, one per page (up to max_pages).
    """
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(
            str(file_path),
            dpi=200,
            poppler_path=POPPLER_PATH,
            first_page=1,
            last_page=max_pages,
        )
        images_b64 = []
        for page in pages:
            import io
            buf = io.BytesIO()
            page.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            images_b64.append(b64)
        print(f"[Vision] Converted {len(images_b64)} page(s) to images")
        return images_b64
    except Exception as e:
        print(f"[Vision] PDF to image conversion failed: {e}")
        return []



# ═════════════════════════════════════════════════════════════════════════════
# CLAUDE VISION — reads handwritten/scanned PDFs directly as images
# Far more accurate than LLaVA for aviation handwriting
# ═════════════════════════════════════════════════════════════════════════════
async def extract_with_claude_vision(images_b64: list[str]) -> dict:
    """
    Send PDF page images directly to Claude Vision API.
    Claude reads handwriting, stamps, signatures, tables — everything.
    Cost: ~$0.02-0.05 per page. Accuracy: far superior to LLaVA.
    Falls back to LLaVA if API key not set.
    """
    if not images_b64:
        return {}

    if not ANTHROPIC_API_KEY:
        print("[Claude Vision] No API key — falling back to LLaVA")
        return {}

    combined: dict = {}

    for i, img_b64 in enumerate(images_b64):
        print(f"[Claude Vision] Processing page {i+1}/{len(images_b64)}...")
        try:
            # Build message with image + text prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": """You are an expert aviation technical records analyst.
Read this aircraft maintenance document image carefully — including all handwritten text, stamps, printed fields, tables, and signatures.

Extract ALL data and return ONLY a valid JSON object with these fields (include only what you can clearly read):
{
  "registration": "aircraft registration e.g. VT-IRB",
  "msn": "manufacturer serial number or AIC Ser No",
  "aircraft_type": "aircraft model",
  "operator": "airline or operator",
  "document_number": "sheet no, work order no, form no, job card no",
  "work_order": "work order or job card number",
  "ata_codes": ["2-digit ATA codes — e.g. 27, 32, 33"],
  "part_numbers": ["all part/component numbers"],
  "serial_numbers": ["all serial numbers — S/N, Ser No, SN"],
  "dates_found": ["all dates visible"],
  "personnel": ["AME names, inspector names, engineer names, coordinator"],
  "licence_numbers": ["AME licence, stamp codes, auth numbers, IGA codes"],
  "maintenance_status": "Serviceable OR Unserviceable OR Defect Found OR Completed",
  "defect_description": "symptom description — what was wrong with the aircraft",
  "rectification": "corrective action — what was done to fix it",
  "next_due": "next maintenance due",
  "document_type_detected": "Work Order / CRS / Maintenance Log / Inspection Report",
  "part_numbers": ["ALL part numbers found anywhere in document"],
  "serial_numbers": ["ALL serial numbers found — SN, S/N, Ser No"],
  "removed_part_numbers": ["part numbers of components that were REMOVED"],
  "removed_serial_numbers": ["serial numbers of components that were REMOVED"],
  "installed_part_numbers": ["part numbers of components that were INSTALLED or FITTED"],
  "installed_serial_numbers": ["serial numbers of components that were INSTALLED or FITTED"],
  "pilot_name": "name of pilot who reported the snag or defect",
  "pilot_reported_snag": "exact snag or defect as reported by pilot — copy from Symptom field verbatim",
  "engineer_name": "name of AME or engineer who performed maintenance work",
  "technician_name": "name of technician who assisted",
  "authorisation_number": "engineer or AME authorisation or approval reference number",
  "qc_inspector": "QC inspector name",
  "qc_id": "QC inspector ID number",
  "maintenance_action": "exact maintenance work done by engineer — copy from Action/Work Done section",
  "airworthiness_directives": ["AD numbers — Airworthiness Directive references"],
  "service_bulletins": ["SB numbers — Service Bulletin references"],
  "flight_hours": "total flight hours if shown",
  "flight_cycles": "total cycles if shown",
  "station": "airport or station ICAO code",
  "work_centre": "work centre code",
  "flight_reference": "flight number if mentioned",
  "signatures_detected": true or false,
  "raw_text": "transcribe ALL text visible in the document exactly as written"
}

CRITICAL RULES:
1. Only extract what you can clearly READ in this image
2. Never invent or guess values not visible in the document
3. Read ALL handwritten text carefully — it is the most important data
4. Include work centre codes, serial numbers from tables, inspector stamps
5. Return ONLY the JSON object — no explanation, no markdown

Read every word in the image now and extract all data."""
                        }
                    ],
                }
            ]

            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    ANTHROPIC_API_URL,
                    headers={
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                        "x-api-key": ANTHROPIC_API_KEY,
                    },
                    json={
                        "model": CLAUDE_MODEL,
                        "max_tokens": 2000,
                        "messages": messages,
                    },
                )

                if resp.status_code != 200:
                    print(f"[Claude Vision] API error {resp.status_code}: {resp.text[:200]}")
                    continue

                result = resp.json()
                raw = result["content"][0]["text"].strip()
                raw = re.sub(r"^```(?:json)?", "", raw).strip()
                raw = re.sub(r"```$", "", raw).strip()

                try:
                    page_data = json.loads(raw)
                except json.JSONDecodeError:
                    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                    if json_match:
                        page_data = json.loads(json_match.group())
                    else:
                        print(f"[Claude Vision] Page {i+1} non-JSON response")
                        continue

                # Merge pages
                for key, val in page_data.items():
                    if val is None or val == "" or val == [] or val == {}:
                        continue
                    if key not in combined:
                        combined[key] = val
                    elif isinstance(val, list) and isinstance(combined.get(key), list):
                        combined[key] = list(dict.fromkeys(combined[key] + val))

                field_count = sum(1 for v in page_data.values() if v)
                print(f"[Claude Vision] Page {i+1} extracted {field_count} fields")

        except Exception as e:
            print(f"[Claude Vision] Page {i+1} failed: {e}")
            continue

    if combined:
        combined["_source"] = "claude_vision"
        combined["_method"] = "claude_vision_api"
        total = sum(1 for v in combined.values() if v and not str(v).startswith("_"))
        print(f"[Claude Vision] Total: {total} fields extracted")
    else:
        print("[Claude Vision] No data extracted")

    return combined

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — LLAVA VISION EXTRACTION (handwritten/scanned PDFs)
# ═════════════════════════════════════════════════════════════════════════════
async def extract_with_llava(images_b64: list[str]) -> dict:
    """
    Send PDF page images to LLaVA vision model.
    LLaVA can READ handwriting directly from images.
    Combines results from all pages into one structured dict.
    """
    if not images_b64:
        return {"_source": "llava_no_images"}

    prompt = """You are an expert aviation technical records analyst. 
Look at this aircraft maintenance document image carefully and extract ALL visible information.

Return ONLY a valid JSON object with these fields (include only what you can actually read):
{
  "registration": "aircraft registration e.g. VT-ABC",
  "msn": "manufacturer serial number",
  "aircraft_type": "e.g. Boeing 737-800",
  "operator": "airline or operator name",
  "document_number": "form or reference number",
  "work_order": "work order or job card number",
  "part_numbers": ["array of part numbers"],
  "serial_numbers": ["array of serial numbers"],
  "ata_codes": ["array of 2-digit ATA codes e.g. 32, 27"],
  "dates_found": ["array of all dates visible"],
  "licence_numbers": ["array of AME or engineer licence numbers"],
  "personnel": ["array of engineer/technician/inspector names"],
  "maintenance_status": "Serviceable OR Unserviceable OR Defect Found OR Completed",
  "defect_description": "description of any defect or finding",
  "rectification": "corrective action taken",
  "next_due": "next due date or flight hours",
  "document_type_detected": "CRS / Work Order / AD Compliance / SB Compliance / Maintenance Log / Inspection Report",
  "airworthiness_directives": ["array of AD numbers"],
  "service_bulletins": ["array of SB numbers"],
  "flight_hours": "total flight hours if visible",
  "flight_cycles": "total flight cycles if visible",
  "station": "airport or station code",
  "signatures_detected": true or false,
  "raw_text": "transcribe ALL visible text from the document exactly as written"
}

Read every word carefully including handwritten text. Return ONLY the JSON, no explanation."""

    combined: dict = {}

    async with httpx.AsyncClient(timeout=300, limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)) as client:
        for i, img_b64 in enumerate(images_b64):
            print(f"[LLaVA] Processing page {i+1}/{len(images_b64)}...")
            # Try primary URL first, then fallback
            urls_to_try = [OLLAMA_URL, OLLAMA_URL_FALLBACK]
            resp = None
            for url in urls_to_try:
                try:
                    resp = await client.post(
                        f"{url}/api/generate",
                        json={
                            "model": OLLAMA_VISION_MODEL,
                            "prompt": prompt,
                            "images": [img_b64],
                            "stream": False,
                            "format": "json",
                            "options": {
                                "temperature": 0.1,
                                "num_predict": 2000,
                            },
                        },
                    )
                    print(f"[LLaVA] Connected via {url}")
                    break
                except Exception as e:
                    print(f"[LLaVA] Failed to connect via {url}: {e}")
                    continue

            if resp is None:
                print(f"[LLaVA] Page {i+1} failed: All connection attempts failed")
                continue
            try:
                result = resp.json()
                raw = result.get("response", "").strip()
                raw = re.sub(r"^```(?:json)?", "", raw).strip()
                raw = re.sub(r"```$", "", raw).strip()

                try:
                    page_data = json.loads(raw)
                except json.JSONDecodeError:
                    # Try to extract JSON from mixed response
                    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                    if json_match:
                        page_data = json.loads(json_match.group())
                    else:
                        print(f"[LLaVA] Page {i+1} returned non-JSON, skipping")
                        continue

                # Merge pages — arrays get combined, strings use first non-empty value
                for key, val in page_data.items():
                    if val is None or val == "" or val == [] or val == {}:
                        continue
                    if key not in combined:
                        combined[key] = val
                    elif isinstance(val, list) and isinstance(combined[key], list):
                        # Merge arrays, deduplicate
                        combined[key] = list(dict.fromkeys(combined[key] + val))
                    elif isinstance(val, str) and isinstance(combined[key], str):
                        # Keep existing string value (first page wins)
                        pass

                print(f"[LLaVA] Page {i+1} extracted {len(page_data)} fields")

            except Exception as e:
                print(f"[LLaVA] Page {i+1} failed: {e}")
                continue

    if combined:
        combined["_source"] = "llava"
        return combined
    else:
        return {"_source": "llava_failed"}


# ═════════════════════════════════════════════════════════════════════════════
# REGEX EXTRACTION (supplement for typed PDFs)
# ═════════════════════════════════════════════════════════════════════════════
def extract_structured_data_regex(text: str) -> dict:
    if not text:
        return {"_source": "regex"}

    data: dict = {}
    t = text
    u = text.upper()

    # ── Registration ─────────────────────────────────────────────────────────
    for p in [
        r"\bA/?C\s+Reg(?:istration)?\s*[:#]?\s*([A-Z]{1,2}-[A-Z0-9]{3,6})\b",
        r"\bRegistration\s*[:#]?\s*([A-Z]{1,2}-[A-Z0-9]{3,6})\b",
        r"\bReg(?:istration)?\s*[:#]?\s*([A-Z]{1,2}-[A-Z0-9]{3,6})\b",
        r"\bA/?C\s+Reg\s*[:#]?\s*([A-Z0-9\-]{4,8})\b",
        r"\b([A-Z]{1,2}-[A-Z0-9]{3,6})\b",
    ]:
        m = re.search(p, t, re.IGNORECASE)
        if m: data["registration"] = m.group(1).strip().upper(); break

    # ── MSN / Serial ─────────────────────────────────────────────────────────
    for p in [
        r"\bMSN\s*[:#]?\s*([A-Z0-9\-]{3,15})\b",
        r"\bA/?C\s+Serial\s+No\.?\s*[:#]?\s*([A-Z0-9\-]{3,15})\b",
        r"\bSheet[\/\s]*Serial\s+No\.?\s*[:#]?\s*([A-Z0-9\-]{4,15})\b",
        r"\bManufacturer(?:'s)?\s+Serial\s+(?:No\.?|Number)\s*[:#]?\s*([A-Z0-9\-]{3,12})\b",
        r"\bSerial\s+No\.?\s*[:#]?\s*([A-Z0-9\-]{3,12})\b",
    ]:
        m = re.search(p, t, re.IGNORECASE)
        if m: data["msn"] = m.group(1).strip(); break

    # ── Aircraft Type ─────────────────────────────────────────────────────────
    for p in [
        r"\bA/?C\s+Type\s*[:#]?\s*([A-Za-z0-9][\w\s\-\/]{2,25}?)(?:\n|$|,|\|)",
        r"\bAircraft\s+(?:Type|Model)\s*[:#]?\s*([A-Za-z0-9][\w\s\-\/]{2,25}?)(?:\n|$|,|\|)",
        r"\b(Boeing\s+\d{3}[\w\-]*)\b",
        r"\b(Airbus\s+A\d{3}[\w\-]*)\b",
        r"\b(ATR[\s\-]\d{2}[\w\-]*)\b",
        r"\b(Embraer\s+(?:E|ERJ)?[\d\w\-]{2,10})\b",
        r"\b(A\d{3}[\-]\d{3})\b",
    ]:
        m = re.search(p, t, re.IGNORECASE)
        if m: data["aircraft_type"] = m.group(1).strip(); break

    # ── Operator ─────────────────────────────────────────────────────────────
    for p in [
        r"\bOriginating\s+Unit\s*[:#]?\s*([A-Z][A-Z0-9\-]{2,20})\b",
        r"\bOperator\s*[:#]?\s*([A-Za-z][\w\s&\.\-]{2,40}?)(?:\n|$|,|\|)",
        r"\bAirline\s*[:#]?\s*([A-Za-z][\w\s&\.\-]{2,40}?)(?:\n|$|,|\|)",
        r"\bCustomer\s*[:#]?\s*([A-Za-z][\w\s&\.\-]{2,40}?)(?:\n|$|,|\|)",
        r"\bCarrier\s*[:#]?\s*([A-Za-z][\w\s&\.\-]{2,40}?)(?:\n|$|,|\|)",
    ]:
        m = re.search(p, t, re.IGNORECASE)
        if m: data["operator"] = m.group(1).strip(); break

    # ── Work Order / Job Card / JCN ───────────────────────────────────────────
    for p in [
        r"\bJCN\s*[:#]?\s*([A-Z0-9\-]{4,20})\b",
        r"\bMWO\s*[:#]?\s*([A-Z0-9\-]{4,20})\b",
        r"\b(?:Work\s*Order|W\.?O\.?)\s*(?:No\.?|#)?\s*[:#]?\s*([A-Z0-9\-]{4,20})\b",
        r"\b(?:Job\s*Card|J\.?C\.?)\s*(?:No\.?|#)?\s*[:#]?\s*([A-Z0-9\-]{4,20})\b",
        r"\bSheet[\/\s]*Serial\s+No\.?\s*[:#]?\s*([A-Z0-9\-]{4,15})\b",
        r"\bTask\s+(?:No\.?|#)\s*[:#]?\s*([A-Z0-9\-]{4,20})\b",
        r"\bLIS\s+JCN\s*[:#]?\s*([A-Z0-9\-]{4,20})\b",
    ]:
        m = re.search(p, t, re.IGNORECASE)
        if m: data["work_order"] = m.group(1).strip(); break

    # ── Document Number ───────────────────────────────────────────────────────
    for p in [
        r"\bDoc(?:ument)?\s*(?:No\.?|#|Ref\.?)\s*[:#]?\s*([A-Z0-9\-\/]{4,20})\b",
        r"\bForm\s*(?:No\.?|#)\s*[:#]?\s*([A-Z0-9\-\/]{4,20})\b",
        r"\bRef(?:erence)?\s*(?:No\.?|#)\s*[:#]?\s*([A-Z0-9\-\/]{4,20})\b",
        r"\bCRS\s*(?:No\.?|#)\s*[:#]?\s*([A-Z0-9\-\/]{4,20})\b",
    ]:
        m = re.search(p, t, re.IGNORECASE)
        if m: data["document_number"] = m.group(1).strip(); break

    # ── Part Numbers ─────────────────────────────────────────────────────────
    pn = re.findall(
        r"\b(?:Part\s*(?:No\.?|Number|#)|P\.?N\.?|Action\s+Prefix)\s*[:#]?\s*([A-Z0-9][\w\-]{3,20})\b",
        t, re.IGNORECASE)
    if pn: data["part_numbers"] = list(dict.fromkeys(p.strip() for p in pn))[:8]

    # ── Serial Numbers ────────────────────────────────────────────────────────
    sn = re.findall(
        r"\b(?:Serial\s*(?:No\.?|Number|#)|S\.?N\.?|Main\s+Equipment.*?Serial\s+No\.?)\s*[:#]?\s*([A-Z0-9][\w\-]{3,20})\b",
        t, re.IGNORECASE)
    sn2 = re.findall(r"\bSN[\-\s]([A-Z0-9]{3,15})\b", t, re.IGNORECASE)
    all_sn = list(dict.fromkeys([s.strip() for s in sn + sn2]))
    if all_sn: data["serial_numbers"] = all_sn[:8]

    # ── Removed Part / Serial Numbers ─────────────────────────────────────────
    rem_pn = re.findall(
        r"\bRemov(?:ed|al)\s+(?:Part\s*(?:No\.?|#)|P\.?N\.?)\s*[:#]?\s*([A-Z0-9][\w\-]{3,20})\b",
        t, re.IGNORECASE)
    rem_pn2 = re.findall(
        r"\bRemov(?:ed|al)[^\n]{0,30}?\bP\.?N\.?\s*[:#]?\s*([A-Z0-9][\w\-]{3,20})\b",
        t, re.IGNORECASE)
    if rem_pn or rem_pn2:
        data["removed_part_numbers"] = list(dict.fromkeys(rem_pn + rem_pn2))[:5]

    rem_sn = re.findall(
        r"\bRemov(?:ed|al)\s+(?:Serial\s*(?:No\.?|#)|S\.?N\.?)\s*[:#]?\s*([A-Z0-9][\w\-]{3,20})\b",
        t, re.IGNORECASE)
    rem_sn2 = re.findall(
        r"\bRemov(?:ed|al)[^\n]{0,30}?\bS\.?N\.?\s*[:#]?\s*([A-Z0-9][\w\-]{3,20})\b",
        t, re.IGNORECASE)
    if rem_sn or rem_sn2:
        data["removed_serial_numbers"] = list(dict.fromkeys(rem_sn + rem_sn2))[:5]

    # ── Installed Part / Serial Numbers ───────────────────────────────────────
    ins_pn = re.findall(
        r"\bInstall(?:ed|ation)\s+(?:Part\s*(?:No\.?|#)|P\.?N\.?)\s*[:#]?\s*([A-Z0-9][\w\-]{3,20})\b",
        t, re.IGNORECASE)
    ins_pn2 = re.findall(
        r"\bInstall(?:ed|ation)[^\n]{0,30}?\bP\.?N\.?\s*[:#]?\s*([A-Z0-9][\w\-]{3,20})\b",
        t, re.IGNORECASE)
    if ins_pn or ins_pn2:
        data["installed_part_numbers"] = list(dict.fromkeys(ins_pn + ins_pn2))[:5]

    ins_sn = re.findall(
        r"\bInstall(?:ed|ation)\s+(?:Serial\s*(?:No\.?|#)|S\.?N\.?)\s*[:#]?\s*([A-Z0-9][\w\-]{3,20})\b",
        t, re.IGNORECASE)
    ins_sn2 = re.findall(
        r"\bInstall(?:ed|ation)[^\n]{0,30}?\bS\.?N\.?\s*[:#]?\s*([A-Z0-9][\w\-]{3,20})\b",
        t, re.IGNORECASE)
    if ins_sn or ins_sn2:
        data["installed_serial_numbers"] = list(dict.fromkeys(ins_sn + ins_sn2))[:5]

    # ── Pilot Name ────────────────────────────────────────────────────────────
    pilot = re.search(
        r"\b(?:Pilot|Capt\.?|Captain|PIC|Reported\s+By|Capt\.?\s+Name)\s*[:#]?\s*([A-Za-z][A-Za-z\s\.]{2,35}?)(?:\n|$|,|;|\|)",
        t, re.IGNORECASE)
    if pilot: data["pilot_name"] = pilot.group(1).strip()

    # ── Pilot Reported Snag ───────────────────────────────────────────────────
    for p in [
        r"\bPilot\s+Report(?:ed)?\s*[:#]?\s*(.{10,300}?)(?:\n\n|\Z)",
        r"\bSymptom\s*[:#]?\s*(.{10,300}?)(?:\n\n|Action|Work\s+Done|\Z)",
        r"\bSnag\s+Report(?:ed)?\s*[:#]?\s*(.{10,300}?)(?:\n\n|\Z)",
        r"\bPilot\s+Complaint\s*[:#]?\s*(.{10,300}?)(?:\n\n|\Z)",
        r"\bDiscrepancy\s*/\s*Problem\s*[:#]?\s*(.{10,300}?)(?:\n\n|\Z)",
    ]:
        m = re.search(p, t, re.IGNORECASE | re.DOTALL)
        if m:
            txt = m.group(1).strip().replace("\n", " ")
            if len(txt) > 5: data["pilot_reported_snag"] = txt[:400]; break

    # ── Engineer Name ─────────────────────────────────────────────────────────
    for p in [
        r"\bAME\s+Name\s*[:#]?\s*([A-Za-z][A-Za-z\s\.]{2,35}?)(?:\n|$|Lic|Auth)",
        r"\bEngineer\s+Name\s*[:#]?\s*([A-Za-z][A-Za-z\s\.]{2,35}?)(?:\n|$|,)",
        r"\bCertified\s+By\s*[:#]?\s*([A-Za-z][A-Za-z\s\.]{2,35}?)(?:\n|$|,)",
        r"\bReleased\s+By\s*[:#]?\s*([A-Za-z][A-Za-z\s\.]{2,35}?)(?:\n|$|,)",
    ]:
        m = re.search(p, t, re.IGNORECASE)
        if m: data["engineer_name"] = m.group(1).strip(); break

    # ── Technician Name ───────────────────────────────────────────────────────
    tech_name = re.search(
        r"\b(?:Technician|Tech\.?|Helper)\s+Name\s*[:#]?\s*([A-Za-z][A-Za-z\s\.]{2,35}?)(?:\n|$|,)",
        t, re.IGNORECASE)
    if tech_name: data["technician_name"] = tech_name.group(1).strip()

    # ── Authorisation Number ──────────────────────────────────────────────────
    for p in [
        r"\bAuth(?:orisation|orization)?\s*(?:No\.?|#|Ref\.?)\s*[:#]?\s*([A-Z0-9][A-Z0-9\/\-]{3,25})\b",
        r"\bApproval\s*(?:No\.?|#)\s*[:#]?\s*([A-Z0-9][A-Z0-9\/\-]{3,25})\b",
        r"\bAMO\s*[:#]?\s*([A-Z0-9\/\-]{4,25})\b",
        r"\bINDIGO\/F\-APP\/([A-Z0-9\/\-]{4,20})\b",
    ]:
        m = re.search(p, t, re.IGNORECASE)
        if m: data["authorisation_number"] = m.group(1).strip(); break

    # ── QC Inspector ──────────────────────────────────────────────────────────
    qc = re.search(
        r"\bQC\s+Inspector\s*[:#]?\s*([A-Za-z][A-Za-z\s\.]{2,35}?)(?:\n|$|\s+ID)",
        t, re.IGNORECASE)
    if qc: data["qc_inspector"] = qc.group(1).strip()

    qcid = re.search(r"\bQC\s*(?:ID|No\.?)\s*[:#]?\s*([A-Z0-9\-]{3,15})\b", t, re.IGNORECASE)
    if qcid: data["qc_id"] = qcid.group(1).strip()

    # ── Maintenance Action ────────────────────────────────────────────────────
    for p in [
        r"\bAction\s*/\s*Work\s+Done\s*[:#]?\s*(.{10,400}?)(?:\n\n|\Z)",
        r"\bWork\s+Done\s*[:#]?\s*(.{10,400}?)(?:\n\n|\Z)",
        r"\bMaintenance\s+(?:Action|Performed|Done)\s*[:#]?\s*(.{10,400}?)(?:\n\n|\Z)",
        r"\bCorrective\s+Action\s+Taken\s*[:#]?\s*(.{10,400}?)(?:\n\n|\Z)",
    ]:
        m = re.search(p, t, re.IGNORECASE | re.DOTALL)
        if m:
            txt = m.group(1).strip().replace("\n", " ")
            if len(txt) > 10: data["maintenance_action"] = txt[:500]; break

    # ── SB References ────────────────────────────────────────────────────────
    sb_raw = re.findall(
        r"\b(?:SB|Service\s+Bulletin)\s*[:#]?\s*([A-Z0-9][\w\-]{3,20})\b",
        t, re.IGNORECASE)
    if sb_raw: data["service_bulletins"] = list(dict.fromkeys(sb_raw))[:5]

    # ── AD References ─────────────────────────────────────────────────────────
    ad_raw2 = re.findall(
        r"\b(?:AD|Airworthiness\s+Directive)\s*[:#]?\s*(\d{4}[\-\/]\d{2,3}[\-\/]\d{2,3})\b",
        t, re.IGNORECASE)
    if ad_raw2: data["airworthiness_directives"] = list(dict.fromkeys(ad_raw2))[:5]

    # ── ATA Codes ─────────────────────────────────────────────────────────────
    ata_raw = re.findall(r"\bATA[\s\-:]*(\d{2})(?:[\-\.](\d{2,4}))?\b", t, re.IGNORECASE)
    # Also catch FCTL-27, FCTL-32 style references
    fctl = re.findall(r"\bF?CTL[\-\s](\d{2})\b", t, re.IGNORECASE)
    all_ata = sorted(set(
        [g[0] for g in ata_raw if g[0] in ATA_MAP] +
        [f for f in fctl if f in ATA_MAP]
    ))
    if all_ata: data["ata_codes"] = all_ata

    # ── ADs ───────────────────────────────────────────────────────────────────
    ad_raw = re.findall(
        r"\b(?:AD|Airworthiness\s+Directive)\s*[:#]?\s*(\d{4}[\-\/]\d{2,3}[\-\/]\d{2,3})\b",
        t, re.IGNORECASE)
    if ad_raw: data["airworthiness_directives"] = list(dict.fromkeys(ad_raw))[:5]

    # ── SBs ───────────────────────────────────────────────────────────────────
    sb_raw = re.findall(
        r"\b(?:SB|Service\s+Bulletin)\s*[:#]?\s*([A-Z0-9][\w\-]{4,20})\b",
        t, re.IGNORECASE)
    if sb_raw: data["service_bulletins"] = list(dict.fromkeys(sb_raw))[:5]

    # ── Dates ─────────────────────────────────────────────────────────────────
    date_raw = re.findall(
        r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
        r"|\d{4}[\/\-]\d{2}[\/\-]\d{2}"
        r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})\b",
        t, re.IGNORECASE)
    if date_raw: data["dates_found"] = list(dict.fromkeys(date_raw))[:6]

    # ── Next Due ──────────────────────────────────────────────────────────────
    for p in [
        r"\bNext\s+Due\s*[:#]?\s*([^\n,;]{3,30})",
        r"\bExpir(?:y|es?|ation)\s+(?:Date\s*)?[:#]?\s*([^\n,;]{3,25})",
        r"\bValid\s+(?:Until|To|Through)\s*[:#]?\s*([^\n,;]{3,25})",
        r"\bDue\s+Date\s*[:#]?\s*([^\n,;]{3,25})",
    ]:
        m = re.search(p, t, re.IGNORECASE)
        if m: data["next_due"] = m.group(1).strip(); break

    # ── Flight Hours / Cycles ─────────────────────────────────────────────────
    fh = re.search(
        r"\b(?:A/?F\s+Hrs?|Total\s+FH|Flight\s+Hours?|TSN|FH)\s*[:#]?\s*([\d,\.]+)\b",
        t, re.IGNORECASE)
    if fh: data["flight_hours"] = fh.group(1).replace(",","").strip()

    fc = re.search(
        r"\b(?:Total\s+FC|Flight\s+Cycles?|Landings?|TSL|FC)\s*[:#]?\s*([\d,]+)\b",
        t, re.IGNORECASE)
    if fc: data["flight_cycles"] = fc.group(1).replace(",","").strip()

    fhs = re.search(
        r"\b(?:HSN|HSO|Hours?\s+Since\s+(?:New|Overhaul))\s*[:#]?\s*([\d,\.]+)\b",
        t, re.IGNORECASE)
    if fhs: data["hours_since_overhaul"] = fhs.group(1).replace(",","").strip()

    # ── Station / Work Centre ─────────────────────────────────────────────────
    for p in [
        r"\bWork\s+Cent(?:re|er)\s*[:#]?\s*([A-Z][A-Z0-9\-]{2,15})\b",
        r"\bStation\s*[:#]?\s*([A-Z]{3,4})\b",
        r"\bAirport\s*[:#]?\s*([A-Z]{3,4})\b",
        r"\bBase\s*[:#]?\s*([A-Z]{3,4})\b",
    ]:
        m = re.search(p, t, re.IGNORECASE)
        if m: data["station"] = m.group(1).strip().upper(); break

    # ── Personnel — AME Name, QC Inspector, Technician ───────────────────────
    personnel = []
    # AME Name: Ramesh Kumar
    ame = re.findall(r"\bAME\s+Name\s*[:#]?\s*([A-Za-z][A-Za-z\s\.]{2,35}?)(?:\n|$|License|Lic)", t, re.IGNORECASE)
    personnel.extend(ame)
    # QC Inspector: Sandeep Rao
    qc = re.findall(r"\bQC\s+Inspector\s*[:#]?\s*([A-Za-z][A-Za-z\s\.]{2,35}?)(?:\n|$|\s+ID)", t, re.IGNORECASE)
    personnel.extend(qc)
    # General technician/engineer patterns
    tech = re.findall(
        r"\b(?:Technician|Engineer|Mechanic|Inspector|Certifier|Certifying\s+Staff"
        r"|Approved\s+By|Released\s+By|Signed\s+By)\s*[:#]?\s*"
        r"([A-Za-z][A-Za-z\s\.]{2,35}?)(?:\n|$|,|;|\|)",
        t, re.IGNORECASE)
    personnel.extend(tech)
    if personnel:
        data["personnel"] = list(dict.fromkeys(n.strip() for n in personnel if len(n.strip()) > 2))[:5]

    # ── Licence Numbers ───────────────────────────────────────────────────────
    lic = re.findall(
        r"\b(?:Licen[sc]e\s*[:#]?\s*|License\s*[:#]?\s*|AME\s+No\.?\s*[:#]?\s*|Auth(?:orisation)?\s*No\.?\s*[:#]?\s*)"
        r"([A-Z0-9][A-Z0-9\/\-]{3,20})\b",
        t, re.IGNORECASE)
    # Also catch inline "License: IND-4521" patterns
    lic2 = re.findall(r"\bLicen[sc]e\s*[:#]\s*([A-Z0-9\-]{4,20})\b", t, re.IGNORECASE)
    # Catch IDs like QC-2291
    qcid = re.findall(r"\bID\s*[:#]?\s*([A-Z]{2,4}[\-]\d{3,6})\b", t, re.IGNORECASE)
    all_lic = list(dict.fromkeys(lic + lic2 + qcid))
    if all_lic: data["licence_numbers"] = all_lic[:5]

    # ── Defect / Symptom / Fault ──────────────────────────────────────────────
    for p in [
        r"\b(?:Symptom|Fault|Defect|Snag|Discrepancy|Finding|Complaint)\s*(?:NRF\s*\([XY]\)\s*)?(?:Information|Description)?\s*[:#]?\s*(.{10,300}?)(?:\n\n|\n[A-Z0-9]\.|\Z)",
        r"\b(?:Defect|Fault|Snag)\s*[:#]?\s*(.{10,200}?)(?:\n\n|\Z)",
        r"\bDescription\s+of\s+(?:Defect|Fault|Work)\s*[:#]?\s*(.{10,200}?)(?:\n\n|\Z)",
        r"Action\s*/\s*Work\s+Done\s*[:#]?\s*(.{10,300}?)(?:\n\n|\Z)",
    ]:
        m = re.search(p, t, re.IGNORECASE | re.DOTALL)
        if m:
            txt = m.group(1).strip().replace("\n"," ")
            if len(txt) > 10: data["defect_description"] = txt[:400]; break

    # ── Rectification / Action Taken ──────────────────────────────────────────
    for p in [
        r"\b(?:Rectification|Corrective\s+Action|Action\s+Taken|Work\s+Done|Remedy|Repair)\s*[:#]?\s*(.{10,300}?)(?:\n\n|\Z)",
        r"\bInstalled\s+(.{10,200}?)(?:\n|$)",
        r"\bRemoved.*?Installed\s+(.{10,200}?)(?:\n\n|\Z)",
        r"lubrication\s+applied[,\s]+(.{10,200}?)(?:\n|$)",
    ]:
        m = re.search(p, t, re.IGNORECASE | re.DOTALL)
        if m:
            txt = m.group(1).strip().replace("\n"," ")
            if len(txt) > 10: data["rectification"] = txt[:400]; break

    # ── Maintenance Status ────────────────────────────────────────────────────
    l = t.lower()
    if any(x in l for x in ["nil defect","no defect","serviceable","released to service",
                              "aircraft is released","fit for flight","airworthy","i certify"]):
        data["maintenance_status"] = "Serviceable"
    elif any(x in l for x in ["unserviceable"," u/s ","not airworthy","aog"]):
        data["maintenance_status"] = "Unserviceable"
    elif any(x in l for x in ["defect found","fault found","snag found","defect observed","excessive freeplay"]):
        data["maintenance_status"] = "Defect Found"
    elif any(x in l for x in ["work completed","task completed","maintenance completed",
                                "check completed","operational test satisfactory"]):
        data["maintenance_status"] = "Completed"

    # ── Document Type Detection ───────────────────────────────────────────────
    if any(x in u for x in ["CERTIFICATE OF RELEASE","RELEASE TO SERVICE","CRS","EASA FORM 1","FAA 8130","AME RELEASE"]):
        data["document_type_detected"] = "CRS / Release to Service"
    elif any(x in u for x in ["MAINTENANCE WORK ORDER","WORK ORDER","JOB CARD","TASK CARD","MWO"]):
        data["document_type_detected"] = "Work Order / Job Card"
    elif any(x in u for x in ["AIRWORTHINESS DIRECTIVE"," AD ","A.D. COMPLIANCE"]):
        data["document_type_detected"] = "AD Compliance"
    elif any(x in u for x in ["SERVICE BULLETIN"," SB ","S.B. COMPLIANCE"]):
        data["document_type_detected"] = "SB Compliance"
    elif any(x in u for x in ["MAINTENANCE LOG","TECH LOG","TECHNICAL LOG"]):
        data["document_type_detected"] = "Maintenance Log"
    elif any(x in u for x in ["INSPECTION REPORT","BORESCOPE","NDT REPORT"]):
        data["document_type_detected"] = "Inspection Report"

    # ── Signatures ────────────────────────────────────────────────────────────
    data["signatures_detected"] = bool(re.search(
        r"\b(?:Signature|Signed|Sign\s+Here|Authorised\s+Signature|Certify|I\s+certify)\b",
        t, re.IGNORECASE))

    data["_source"] = "regex"
    return data



# ═════════════════════════════════════════════════════════════════════════════
# CLAUDE API EXTRACTION — handles ANY airline format intelligently
# ═════════════════════════════════════════════════════════════════════════════
async def extract_with_claude(text) -> dict:
    """
    Use Claude API to extract structured data from ANY aviation document format.
    Works with IndiGo AMOS, DGCA, JAP100C, EASA, FAA formats automatically.
    Falls back to Ollama if API key not set or call fails.
    """
    # Ensure text is always a clean string — LLaVA sometimes returns dicts
    if isinstance(text, dict):
        text = " ".join(f"{k}: {v}" for k, v in text.items() if v and not str(k).startswith("_"))
    elif isinstance(text, (list, tuple)):
        text = " ".join(str(x) for x in text)
    elif not isinstance(text, str):
        text = str(text)

    text = text.strip()

    if len(text) < 30:
        print("[Claude] Text too short — skipping")
        return {}

    if not ANTHROPIC_API_KEY:
        print("[Claude] No API key set — falling back to Ollama")
        return {}

    prompt = f"""You are an expert aviation technical records analyst with deep knowledge of all airline maintenance documentation formats including DGCA, EASA, FAA, JAP100C, AMOS, TRAX, and all major MRO systems.

Extract ALL structured data from this aircraft maintenance document. This may be from any airline — IndiGo, Air India, SpiceJet, GoAir, Vistara, international carriers — each with different field names for the same data.

Return ONLY a valid JSON object. Include every field you can confidently identify:

{{
  "registration": "aircraft registration e.g. VT-IRB, VT-XYZ",
  "msn": "manufacturer serial number",
  "aircraft_type": "e.g. ATR-72-212, Airbus A320, Boeing 737-800",
  "operator": "airline or operator name",
  "document_number": "work order, form, sheet, or reference number (W/O, WO, MWO, JCN, LIS JCN, Sheet No, Job Card No)",
  "work_order": "same as document_number if it is a work order",
  "iga_code": "IGA or engineer stamp code if present",
  "ata_codes": ["2-digit ATA chapter codes — FCTL-27-30 means 27, CDL means 33 etc"],
  "part_numbers": ["ALL part numbers found anywhere in document"],
  "serial_numbers": ["ALL serial numbers found — SN, S/N, Ser No"],
  "removed_part_numbers": ["part numbers of components that were REMOVED"],
  "removed_serial_numbers": ["serial numbers of components that were REMOVED"],
  "installed_part_numbers": ["part numbers of components that were INSTALLED/FITTED"],
  "installed_serial_numbers": ["serial numbers of components that were INSTALLED/FITTED"],
  "dates_found": ["all dates in document"],
  "pilot_name": "name of pilot who reported the snag/defect",
  "pilot_reported_snag": "exact snag or defect as reported by pilot — verbatim",
  "engineer_name": "name of engineer/AME who did the maintenance work",
  "technician_name": "name of technician who assisted",
  "personnel": ["all names — AME, engineers, inspectors, technicians, pilots, coordinators"],
  "licence_numbers": ["AME licence numbers"],
  "authorisation_number": "engineer/AME authorisation or approval number",
  "iga_code": "IGA stamp code (IndiGo format)",
  "qc_inspector": "QC inspector name",
  "qc_id": "QC inspector ID number",
  "maintenance_status": "Serviceable OR Unserviceable OR Defect Found OR Completed",
  "pilot_reported_snag": "defect/snag as reported by pilot",
  "defect_description": "full symptom/fault/CDL description — what was wrong",
  "maintenance_action": "exact maintenance work performed by engineer",
  "rectification": "corrective action summary — how it was fixed",
  "next_due": "next maintenance due date or hours",
  "document_type_detected": "Work Order / CRS / AD Compliance / SB Compliance / Maintenance Log / Defect Report",
  "airworthiness_directives": ["AD numbers — Airworthiness Directive references"],
  "service_bulletins": ["SB numbers — Service Bulletin references"],
  "flight_hours": "total aircraft flight hours — A/F Hrs, TSN, Total FH",
  "flight_cycles": "total cycles — FC, TSL, Landings, Total Aircraft Cycles",
  "station": "airport or station ICAO code",
  "work_centre": "maintenance work centre code",
  "flight_reference": "flight number if mentioned",
  "signatures_detected": true or false,
  "certification_statement": "AME release to service statement"
}}

CRITICAL RULES — strictly follow these:
1. ONLY extract data that is EXPLICITLY present in the document text below
2. NEVER invent, guess, or hallucinate any field values
3. If a field is not clearly present in the text — OMIT it entirely, do not guess
4. Do NOT fill registration/aircraft type/operator from general knowledge
5. Different airlines use different labels: W/O = WO = MWO = JCN = Sheet No = all mean work order
6. IGA36288 is an engineer stamp code — put in licence_numbers
7. FCTL-27-30 means ATA chapter 27. A/F Hrs = flight hours. Place/Station = airport code
8. If you are not 100% certain a value is in the text, leave that field out

Document text (extract ONLY from this — do not use outside knowledge):
---
{text[:6000]}
---

Return ONLY the JSON object with fields you are CERTAIN about. No explanation. No markdown fences."""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                ANTHROPIC_API_URL,
                headers={{
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                    "x-api-key": ANTHROPIC_API_KEY,
                }},
                json={{
                    "model": CLAUDE_MODEL,
                    "max_tokens": 2000,
                    "messages": [{{"role": "user", "content": prompt}}],
                }},
            )

            if resp.status_code != 200:
                print(f"[Claude] API error {resp.status_code}: {resp.text[:200]}")
                return {}

            result = resp.json()
            raw = result["content"][0]["text"].strip()
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
            parsed = json.loads(raw)
            parsed["_source"] = "claude_ai"
            parsed["_method"] = "claude_api"
            field_count = sum(1 for v in parsed.values() if v and not str(v).startswith("_"))
            print(f"[Claude] Extracted {field_count} fields using {CLAUDE_MODEL}")
            return parsed

    except json.JSONDecodeError as e:
        print(f"[Claude] JSON parse error: {e} — raw: {raw[:200]}")
        return {}
    except Exception as e:
        print(f"[Claude] API call failed: {e}")
        return {}


# ═════════════════════════════════════════════════════════════════════════════
# OLLAMA TEXT EXTRACTION — local llama3.2 for typed PDFs
# ═════════════════════════════════════════════════════════════════════════════
async def extract_with_ollama_text(text: str) -> dict:
    """
    Use local Ollama llama3.2 to extract structured data from typed PDFs.
    Free and runs offline. Falls back to regex if Ollama not running.
    """
    if not text or len(text) < 30:
        return {}

    prompt = f"""You are an aviation technical records analyst. Extract structured data from this aircraft document.

Return ONLY a JSON object with these fields (include only what you find):
{{
  "registration": "aircraft registration",
  "msn": "serial number",
  "aircraft_type": "aircraft model",
  "operator": "airline name",
  "document_number": "work order or form number",
  "work_order": "work order number",
  "ata_codes": ["2-digit ATA codes"],
  "part_numbers": ["part numbers"],
  "serial_numbers": ["serial numbers"],
  "dates_found": ["dates"],
  "personnel": ["engineer/AME names"],
  "licence_numbers": ["licence/ID numbers"],
  "maintenance_status": "Serviceable/Unserviceable/Defect Found/Completed",
  "defect_description": "what was wrong",
  "rectification": "what was done to fix it",
  "next_due": "next due date or hours",
  "document_type_detected": "Work Order/CRS/AD Compliance/SB Compliance/Maintenance Log",
  "flight_hours": "total flight hours",
  "flight_cycles": "total cycles",
  "station": "airport code",
  "signatures_detected": true or false
}}

Document:
---
{text[:4000]}
---

Return ONLY the JSON. No explanation."""

    urls = [OLLAMA_URL, OLLAMA_URL_FALLBACK]
    for url in urls:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{url}/api/generate",
                    json={{
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {{"temperature": 0.1, "num_predict": 1500}},
                    }},
                )
                result = resp.json()
                raw = result.get("response", "").strip()
                raw = re.sub(r"^```(?:json)?", "", raw).strip()
                raw = re.sub(r"```$", "", raw).strip()
                parsed = json.loads(raw)
                parsed["_source"] = "ollama"
                parsed["_method"] = "ollama_text"
                print(f"[Ollama] Extracted {len(parsed)} fields using {OLLAMA_MODEL}")
                return parsed
        except Exception as e:
            print(f"[Ollama] Failed via {url}: {e}")
            continue
    return {}

# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
async def extract_from_pdf(file_path: Path, mode: str = "") -> tuple[str, dict]:
    """
    AeroDoc Extraction Pipeline v6.1
    ─────────────────────────────────
    Dropdown controls everything:

    CLAUDE mode:
      Typed PDF (300+ chars) → Claude text API → regex supplement
      Handwritten/scanned    → Claude Vision API → regex supplement
      Fallback: Ollama → Regex

    OLLAMA mode:
      Typed PDF              → Ollama llama3.2 → regex supplement
      Handwritten/scanned    → LLaVA vision → regex supplement

    LLAVA mode:
      Forces vision on ALL PDFs regardless of text content
      Useful for badly scanned typed docs

    REGEX mode:
      Pure regex on extracted text — no AI at all
      Fastest, free, basic extraction only
    """
    active_mode = mode.strip().lower() if mode.strip() else EXTRACTION_MODE
    print(f"\n[Pipeline] Processing: {file_path.name} | Mode: {active_mode.upper()}")

    # ── REGEX MODE — pure regex, no AI, skip everything else ─────────────────
    if active_mode == "regex":
        extracted_text = extract_text_from_pdf(file_path)
        print(f"[Pipeline] REGEX mode — extracted {len(extracted_text)} chars")
        structured = extract_structured_data_regex(extracted_text)
        structured["_source"] = "regex"
        structured["_method"] = "regex_only"
        print(f"[Pipeline] Regex complete — {sum(1 for v in structured.values() if v and not str(v).startswith('_'))} fields")
        return extracted_text, structured

    # ── LLAVA MODE — force vision on everything ───────────────────────────────
    if active_mode == "llava":
        extracted_text = extract_text_from_pdf(file_path)
        print("[Pipeline] LLAVA mode — forcing vision extraction")
        images_b64 = pdf_to_images_base64(file_path, max_pages=4)
        if not images_b64:
            print("[Pipeline] Image conversion failed → regex fallback")
            structured = extract_structured_data_regex(extracted_text)
            structured["_source"] = "regex_fallback"
            return extracted_text, structured
        structured = await extract_with_llava(images_b64)
        raw_text = structured.pop("raw_text", None)
        if raw_text and len(str(raw_text)) > 50:
            extracted_text = str(raw_text)
        regex_data = extract_structured_data_regex(extracted_text)
        for k, v in regex_data.items():
            if not k.startswith("_") and (k not in structured or not structured[k]):
                structured[k] = v
        structured["_source"] = "llava"
        structured["_method"] = "llava_forced"
        print(f"[Pipeline] LLaVA complete — {sum(1 for v in structured.values() if v and not str(v).startswith('_'))} fields")
        return extracted_text, structured

    # ── CLAUDE / OLLAMA MODE — auto-detect typed vs handwritten ──────────────
    extracted_text = extract_text_from_pdf(file_path)
    text_length = len(extracted_text.strip())
    print(f"[Pipeline] pdfplumber extracted {text_length} characters")

    # 300 chars = enough meaningful text to treat as typed PDF
    # Below 300 = likely handwritten or scanned → use vision
    IS_TYPED = text_length >= 300
    print(f"[Pipeline] Document type: {'TYPED' if IS_TYPED else 'HANDWRITTEN/SCANNED'}")

    if IS_TYPED:
        # ── TYPED PDF ─────────────────────────────────────────────────────────
        structured = {}

        if active_mode == "claude":
            print("[Pipeline] Typed PDF → Claude API")
            structured = await extract_with_claude(extracted_text)
            if not structured:
                print("[Pipeline] Claude failed → Ollama fallback")
                structured = await extract_with_ollama_text(extracted_text)

        elif active_mode == "ollama":
            print("[Pipeline] Typed PDF → Ollama llama3.2")
            structured = await extract_with_ollama_text(extracted_text)

        # Regex always supplements missing fields
        regex_data = extract_structured_data_regex(extracted_text)
        for k, v in regex_data.items():
            if not k.startswith("_") and (k not in structured or not structured[k]):
                structured[k] = v

        if not structured.get("_source"):
            structured["_source"] = "regex"
            structured["_method"] = "regex_only"

        field_count = sum(1 for v in structured.values() if v and not str(v).startswith("_"))
        print(f"[Pipeline] Typed extraction complete — {field_count} fields | source: {structured.get('_source')}")
        return extracted_text, structured

    else:
        # ── HANDWRITTEN / SCANNED PDF ─────────────────────────────────────────
        print("[Pipeline] Handwritten/scanned → converting to images")
        images_b64 = pdf_to_images_base64(file_path, max_pages=4)

        if not images_b64:
            print("[Pipeline] Image conversion failed → regex fallback")
            structured = extract_structured_data_regex(extracted_text)
            structured["_source"] = "regex_fallback"
            structured["_method"] = "image_conversion_failed"
            return extracted_text, structured

        structured = {}

        if active_mode == "claude":
            # Claude Vision reads handwriting directly from image — best accuracy
            print("[Pipeline] Handwritten → Claude Vision API")
            structured = await extract_with_claude_vision(images_b64)
            if not structured:
                print("[Pipeline] Claude Vision failed → LLaVA fallback")
                structured = await extract_with_llava(images_b64)
                raw_text = structured.pop("raw_text", None)
                if raw_text and len(str(raw_text)) > 50:
                    extracted_text = str(raw_text)
                structured["_source"] = "llava"
                structured["_method"] = "llava_fallback"
            else:
                raw_text = structured.pop("raw_text", None)
                if raw_text and len(str(raw_text)) > 50:
                    extracted_text = str(raw_text)
                    print(f"[Pipeline] Claude Vision transcribed {len(extracted_text)} chars")
                structured["_source"] = "claude_vision"
                structured["_method"] = "claude_vision_api"

        elif active_mode == "ollama":
            # LLaVA reads handwriting locally — free
            print("[Pipeline] Handwritten → LLaVA vision (Ollama mode)")
            structured = await extract_with_llava(images_b64)
            raw_text = structured.pop("raw_text", None)
            if raw_text and len(str(raw_text)) > 50:
                extracted_text = str(raw_text)
                print(f"[Pipeline] LLaVA transcribed {len(extracted_text)} chars")
            structured["_source"] = "llava"
            structured["_method"] = "llava_vision"

        # Regex supplements whatever vision found
        if extracted_text:
            regex_data = extract_structured_data_regex(extracted_text)
            for k, v in regex_data.items():
                if not k.startswith("_") and (k not in structured or not structured[k]):
                    structured[k] = v

        field_count = sum(1 for v in structured.values() if v and not str(v).startswith("_"))
        source = structured.get("_source", "unknown")
        print(f"[Pipeline] Vision complete — {field_count} fields | source: {source}")
        return extracted_text, structured


# ═════════════════════════════════════════════════════════════════════════════
# ATA CHAPTER DETECTION
# ═════════════════════════════════════════════════════════════════════════════
def detect_ata_chapter(text: str) -> str:
    if not text: return "unassigned"
    detected = []
    upper = text.upper()
    for code, name in ATA_MAP.items():
        if (f"ATA {code}" in upper or f"ATA-{code}" in upper
                or f"ATA{code}" in upper or f"CHAPTER {code}" in upper):
            detected.append(f"{code} - {name}")
    return ", ".join(detected) if detected else "unassigned"


# ═════════════════════════════════════════════════════════════════════════════
# ROOT
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/", tags=["Root"])
def root():
    claude_ready = bool(ANTHROPIC_API_KEY)
    return {
        "platform": "AeroDoc — Aircraft Technical Records Platform",
        "version": "6.0.0",
        "default_extraction_mode": EXTRACTION_MODE.upper(),
        "per_upload_mode": "supported — select in frontend dropdown",
        "claude_api": "READY" if claude_ready else "NOT CONFIGURED",
        "ollama_text_model": OLLAMA_MODEL,
        "ollama_vision_model": OLLAMA_VISION_MODEL,
        "modes_available": ["claude", "ollama", "llava", "regex"],
        "docs": "/docs",
    }


# ═════════════════════════════════════════════════════════════════════════════
# AIRCRAFT ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/aircraft", response_model=AircraftResponse, tags=["Aircraft"])
def create_aircraft(payload: AircraftCreate, db: Session = Depends(get_db)):
    if db.query(Aircraft).filter(Aircraft.registration == payload.registration).first():
        raise HTTPException(400, "Aircraft with this registration already exists")
    ac = Aircraft(**payload.dict())
    db.add(ac); db.commit(); db.refresh(ac)
    return ac

@app.get("/aircraft", response_model=list[AircraftResponse], tags=["Aircraft"])
def list_aircraft(db: Session = Depends(get_db)):
    return db.query(Aircraft).all()

@app.get("/aircraft/{aircraft_id}", response_model=AircraftResponse, tags=["Aircraft"])
def get_aircraft(aircraft_id: int, db: Session = Depends(get_db)):
    ac = db.query(Aircraft).filter(Aircraft.id == aircraft_id).first()
    if not ac: raise HTTPException(404, "Aircraft not found")
    return ac

@app.put("/aircraft/{aircraft_id}", response_model=AircraftResponse, tags=["Aircraft"])
def update_aircraft(aircraft_id: int, payload: AircraftCreate, db: Session = Depends(get_db)):
    ac = db.query(Aircraft).filter(Aircraft.id == aircraft_id).first()
    if not ac: raise HTTPException(404, "Aircraft not found")
    dup = db.query(Aircraft).filter(
        Aircraft.registration == payload.registration, Aircraft.id != aircraft_id).first()
    if dup: raise HTTPException(400, "Another aircraft with this registration already exists")
    for k, v in payload.dict().items(): setattr(ac, k, v)
    db.commit(); db.refresh(ac)
    return ac

@app.delete("/aircraft/{aircraft_id}", tags=["Aircraft"])
def delete_aircraft(aircraft_id: int, db: Session = Depends(get_db)):
    ac = db.query(Aircraft).filter(Aircraft.id == aircraft_id).first()
    if not ac: raise HTTPException(404, "Aircraft not found")
    db.delete(ac); db.commit()
    return {"message": "Aircraft deleted"}


# ═════════════════════════════════════════════════════════════════════════════
# DOCUMENT ENDPOINTS  (fixed-path routes BEFORE /{document_id})
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/documents/search", response_model=list[DocumentResponse], tags=["Documents"])
def search_documents(query: str = Query(...), db: Session = Depends(get_db)):
    return db.query(Document).filter(Document.extracted_text.ilike(f"%{query}%")).all()

@app.get("/documents/filter/by-chapter", response_model=list[DocumentResponse], tags=["Documents"])
def filter_by_chapter(chapter: str, db: Session = Depends(get_db)):
    return db.query(Document).filter(Document.chapter.contains(chapter)).all()

@app.get("/documents/filter/by-aircraft", response_model=list[DocumentResponse], tags=["Documents"])
def filter_by_aircraft(aircraft_id: int, db: Session = Depends(get_db)):
    return db.query(Document).filter(Document.aircraft_id == aircraft_id).all()

@app.get("/documents/filter/by-type", response_model=list[DocumentResponse], tags=["Documents"])
def filter_by_type(doc_type: str, db: Session = Depends(get_db)):
    return db.query(Document).filter(Document.document_type.ilike(f"%{doc_type}%")).all()

@app.get("/documents/filter/by-ata", response_model=list[DocumentResponse], tags=["Documents"])
def filter_by_ata(ata: str, db: Session = Depends(get_db)):
    docs = db.query(Document).all()
    return [d for d in docs if d.structured_data and isinstance(d.structured_data, dict)
            and ata in d.structured_data.get("ata_codes", [])]

@app.get("/documents/filter/by-status", response_model=list[DocumentResponse], tags=["Documents"])
def filter_by_status(status: str, db: Session = Depends(get_db)):
    docs = db.query(Document).all()
    return [d for d in docs if d.structured_data and isinstance(d.structured_data, dict)
            and d.structured_data.get("maintenance_status", "").lower() == status.lower()]

@app.get("/documents/filter/by-personnel", response_model=list[DocumentResponse], tags=["Documents"])
def filter_by_personnel(name: str, db: Session = Depends(get_db)):
    docs = db.query(Document).all()
    return [d for d in docs if d.structured_data and isinstance(d.structured_data, dict)
            and any(name.lower() in p.lower() for p in d.structured_data.get("personnel", []))]

@app.get("/documents", response_model=list[DocumentResponse], tags=["Documents"])
def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).all()


@app.post("/documents/upload", response_model=DocumentResponse, tags=["Documents"])
async def upload_document(
    aircraft_id:    int  = Form(...),
    document_type:  str  = Form("general"),
    chapter:        str  = Form("unassigned"),
    extraction_mode: str = Form(""),   # overrides global EXTRACTION_MODE if set
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    ac = db.query(Aircraft).filter(Aircraft.id == aircraft_id).first()
    if not ac: raise HTTPException(404, "Aircraft not found")
    if not file.filename: raise HTTPException(400, "File name is missing")

    folder = STORAGE_ROOT / f"aircraft_{aircraft_id}"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / file.filename

    with dest.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    extracted_text  = None
    structured_data: dict = {}

    if file.filename.lower().endswith(".pdf"):
        # Use per-upload extraction mode if provided, else fall back to global
        mode = extraction_mode.strip().lower() if extraction_mode.strip() else EXTRACTION_MODE
        extracted_text, structured_data = await extract_from_pdf(dest, mode=mode)

        # Auto-detect ATA chapter
        if extracted_text:
            chapter = detect_ata_chapter(extracted_text)

        # Auto-fill document type if user left it as "general"
        if document_type == "general" and structured_data.get("document_type_detected"):
            document_type = structured_data["document_type_detected"]

    doc = Document(
        aircraft_id    = aircraft_id,
        file_name      = file.filename,
        file_path      = str(dest),
        document_type  = document_type,
        chapter        = chapter,
        extracted_text = extracted_text,
        structured_data= structured_data,
        uploaded_at    = datetime.utcnow().isoformat(),
    )
    db.add(doc); db.commit(); db.refresh(doc)
    return doc


@app.get("/documents/{document_id}", response_model=DocumentResponse, tags=["Documents"])
def get_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc: raise HTTPException(404, "Document not found")
    return doc

@app.get("/documents/{document_id}/download", tags=["Documents"])
def download_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc: raise HTTPException(404, "Document not found")
    fp = Path(doc.file_path)
    if not fp.exists(): raise HTTPException(404, "File not found on disk")
    return FileResponse(path=str(fp), filename=doc.file_name)

@app.delete("/documents/{document_id}", tags=["Documents"])
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc: raise HTTPException(404, "Document not found")
    fp = Path(doc.file_path)
    db.delete(doc); db.commit()
    if fp.exists(): fp.unlink()
    return {"message": "Document deleted"}

@app.post("/documents/{document_id}/reanalyse", response_model=DocumentResponse, tags=["Documents"])
async def reanalyse_document(
    document_id: int,
    extraction_mode: str = "",
    db: Session = Depends(get_db)
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc: raise HTTPException(404, "Document not found")

    mode = extraction_mode.strip().lower() if extraction_mode.strip() else EXTRACTION_MODE
    fp = Path(doc.file_path)
    if fp.exists() and fp.suffix.lower() == ".pdf":
        # Re-run full pipeline on original file
        extracted_text, structured_data = await extract_from_pdf(fp, mode=mode)
        doc.extracted_text  = extracted_text or doc.extracted_text
        doc.structured_data = structured_data
        doc.chapter         = detect_ata_chapter(doc.extracted_text or "")
    elif doc.extracted_text:
        # No file but have text — run regex
        doc.structured_data = extract_structured_data_regex(doc.extracted_text)
    else:
        raise HTTPException(400, "No file on disk and no extracted text available")

    db.commit(); db.refresh(doc)
    return doc



# ═════════════════════════════════════════════════════════════════════════════
# DOCUMENT REVIEW — PDF preview as JPG + manual field correction
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/documents/{document_id}/preview", tags=["Documents"])
def get_document_preview(document_id: int, page: int = 1, db: Session = Depends(get_db)):
    """
    Convert a PDF page to JPG and return as base64.
    Searches multiple locations if stored path not found.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    if not doc.file_name.lower().endswith(".pdf"):
        raise HTTPException(400, "Preview only available for PDF files")

    # Try multiple possible file locations
    stored_path = Path(doc.file_path)
    candidate_paths = [
        stored_path,
        STORAGE_ROOT / stored_path.parent.name / stored_path.name,
        STORAGE_ROOT / f"aircraft_{doc.aircraft_id}" / doc.file_name,
        Path("storage") / f"aircraft_{doc.aircraft_id}" / doc.file_name,
        Path(doc.file_name),
    ]

    fp = None
    for p in candidate_paths:
        if p.exists():
            fp = p
            print(f"[Preview] Found file at: {fp}")
            break

    if fp is None:
        print(f"[Preview] File not found. Tried: {[str(p) for p in candidate_paths]}")
        raise HTTPException(404,
            f"PDF file not found on disk. "
            f"The file may have been moved or deleted. "
            f"Please re-upload the document to enable preview."
        )

    try:
        from pdf2image import convert_from_path
        import io

        # Get total pages first at low DPI (fast)
        all_pages_info = convert_from_path(
            str(fp), dpi=30, poppler_path=POPPLER_PATH,
        )
        total = len(all_pages_info)
        del all_pages_info  # free memory

        if page < 1 or page > total:
            raise HTTPException(400, f"Page {page} out of range (1-{total})")

        # Get requested page at display DPI
        pages = convert_from_path(
            str(fp),
            dpi=150,
            poppler_path=POPPLER_PATH,
            first_page=page,
            last_page=page,
        )

        if not pages:
            raise HTTPException(500, "Page conversion failed")

        buf = io.BytesIO()
        pages[0].save(buf, format="JPEG", quality=82)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "document_id": document_id,
            "page": page,
            "total_pages": total,
            "image_b64": b64,
            "mime_type": "image/jpeg",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Preview] Error: {e}")
        raise HTTPException(500, f"Preview generation failed: {e}")


@app.patch("/documents/{document_id}/correct", response_model=DocumentResponse, tags=["Documents"])
def correct_document_fields(
    document_id: int,
    corrections: dict,
    db: Session = Depends(get_db),
):
    """
    Manually correct extracted fields for a document.
    Accepts a dict of field corrections and merges into structured_data.
    Also updates document_type and chapter if provided.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    # Update top-level document fields if provided
    if "document_type" in corrections:
        doc.document_type = corrections.pop("document_type")
    if "chapter" in corrections:
        doc.chapter = corrections.pop("chapter")

    # Merge remaining corrections into structured_data
    current = dict(doc.structured_data or {})
    for key, val in corrections.items():
        if val is None or val == "":
            current.pop(key, None)   # remove empty fields
        else:
            current[key] = val

    # Mark as manually corrected
    current["_manually_corrected"] = True
    current["_corrected_at"] = datetime.utcnow().isoformat()

    doc.structured_data = current
    db.commit()
    db.refresh(doc)
    return doc

# ═════════════════════════════════════════════════════════════════════════════
# AIRWORTHINESS DIRECTIVES
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/ads", response_model=ADResponse, tags=["Airworthiness"])
def create_ad(payload: ADCreate, db: Session = Depends(get_db)):
    if not db.query(Aircraft).filter(Aircraft.id == payload.aircraft_id).first():
        raise HTTPException(404, "Aircraft not found")
    ad = AirworthinessDirective(**payload.dict())
    db.add(ad); db.commit(); db.refresh(ad); return ad

@app.get("/ads", response_model=list[ADResponse], tags=["Airworthiness"])
def list_ads(aircraft_id: int|None=None, status: str|None=None, db: Session=Depends(get_db)):
    q = db.query(AirworthinessDirective)
    if aircraft_id: q = q.filter(AirworthinessDirective.aircraft_id == aircraft_id)
    if status:      q = q.filter(AirworthinessDirective.status == status)
    return q.all()

@app.get("/ads/{ad_id}", response_model=ADResponse, tags=["Airworthiness"])
def get_ad(ad_id: int, db: Session = Depends(get_db)):
    ad = db.query(AirworthinessDirective).filter(AirworthinessDirective.id == ad_id).first()
    if not ad: raise HTTPException(404, "AD not found"); return ad

@app.put("/ads/{ad_id}", response_model=ADResponse, tags=["Airworthiness"])
def update_ad(ad_id: int, payload: ADCreate, db: Session = Depends(get_db)):
    ad = db.query(AirworthinessDirective).filter(AirworthinessDirective.id == ad_id).first()
    if not ad: raise HTTPException(404, "AD not found")
    for k, v in payload.dict().items(): setattr(ad, k, v)
    db.commit(); db.refresh(ad); return ad

@app.delete("/ads/{ad_id}", tags=["Airworthiness"])
def delete_ad(ad_id: int, db: Session = Depends(get_db)):
    ad = db.query(AirworthinessDirective).filter(AirworthinessDirective.id == ad_id).first()
    if not ad: raise HTTPException(404, "AD not found")
    db.delete(ad); db.commit(); return {"message": "AD deleted"}


# ═════════════════════════════════════════════════════════════════════════════
# SERVICE BULLETINS
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/sbs", response_model=SBResponse, tags=["Service Bulletins"])
def create_sb(payload: SBCreate, db: Session = Depends(get_db)):
    if not db.query(Aircraft).filter(Aircraft.id == payload.aircraft_id).first():
        raise HTTPException(404, "Aircraft not found")
    sb = ServiceBulletin(**payload.dict())
    db.add(sb); db.commit(); db.refresh(sb); return sb

@app.get("/sbs", response_model=list[SBResponse], tags=["Service Bulletins"])
def list_sbs(aircraft_id: int|None=None, status: str|None=None, db: Session=Depends(get_db)):
    q = db.query(ServiceBulletin)
    if aircraft_id: q = q.filter(ServiceBulletin.aircraft_id == aircraft_id)
    if status:      q = q.filter(ServiceBulletin.status == status)
    return q.all()

@app.get("/sbs/{sb_id}", response_model=SBResponse, tags=["Service Bulletins"])
def get_sb(sb_id: int, db: Session = Depends(get_db)):
    sb = db.query(ServiceBulletin).filter(ServiceBulletin.id == sb_id).first()
    if not sb: raise HTTPException(404, "SB not found"); return sb

@app.put("/sbs/{sb_id}", response_model=SBResponse, tags=["Service Bulletins"])
def update_sb(sb_id: int, payload: SBCreate, db: Session = Depends(get_db)):
    sb = db.query(ServiceBulletin).filter(ServiceBulletin.id == sb_id).first()
    if not sb: raise HTTPException(404, "SB not found")
    for k, v in payload.dict().items(): setattr(sb, k, v)
    db.commit(); db.refresh(sb); return sb

@app.delete("/sbs/{sb_id}", tags=["Service Bulletins"])
def delete_sb(sb_id: int, db: Session = Depends(get_db)):
    sb = db.query(ServiceBulletin).filter(ServiceBulletin.id == sb_id).first()
    if not sb: raise HTTPException(404, "SB not found")
    db.delete(sb); db.commit(); return {"message": "SB deleted"}


# ═════════════════════════════════════════════════════════════════════════════
# MAINTENANCE CHECKS
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/checks", response_model=CheckResponse, tags=["Maintenance"])
def create_check(payload: CheckCreate, db: Session = Depends(get_db)):
    if not db.query(Aircraft).filter(Aircraft.id == payload.aircraft_id).first():
        raise HTTPException(404, "Aircraft not found")
    chk = MaintenanceCheck(**payload.dict())
    db.add(chk); db.commit(); db.refresh(chk); return chk

@app.get("/checks", response_model=list[CheckResponse], tags=["Maintenance"])
def list_checks(aircraft_id: int|None=None, status: str|None=None, db: Session=Depends(get_db)):
    q = db.query(MaintenanceCheck)
    if aircraft_id: q = q.filter(MaintenanceCheck.aircraft_id == aircraft_id)
    if status:      q = q.filter(MaintenanceCheck.status == status)
    return q.all()

@app.get("/checks/{check_id}", response_model=CheckResponse, tags=["Maintenance"])
def get_check(check_id: int, db: Session = Depends(get_db)):
    chk = db.query(MaintenanceCheck).filter(MaintenanceCheck.id == check_id).first()
    if not chk: raise HTTPException(404, "Check not found"); return chk

@app.put("/checks/{check_id}", response_model=CheckResponse, tags=["Maintenance"])
def update_check(check_id: int, payload: CheckCreate, db: Session = Depends(get_db)):
    chk = db.query(MaintenanceCheck).filter(MaintenanceCheck.id == check_id).first()
    if not chk: raise HTTPException(404, "Check not found")
    for k, v in payload.dict().items(): setattr(chk, k, v)
    db.commit(); db.refresh(chk); return chk

@app.delete("/checks/{check_id}", tags=["Maintenance"])
def delete_check(check_id: int, db: Session = Depends(get_db)):
    chk = db.query(MaintenanceCheck).filter(MaintenanceCheck.id == check_id).first()
    if not chk: raise HTTPException(404, "Check not found")
    db.delete(chk); db.commit(); return {"message": "Check deleted"}


# ═════════════════════════════════════════════════════════════════════════════
# STATS
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/stats/summary", tags=["Stats"])
def get_summary(db: Session = Depends(get_db)):
    today = date.today().isoformat()
    in30  = date.today().replace(day=min(date.today().day+30,28)).isoformat()
    all_docs = db.query(Document).all()
    all_ads  = db.query(AirworthinessDirective).all()
    all_chks = db.query(MaintenanceCheck).all()

    overdue_ads  = sum(1 for a in all_ads if a.due_date and a.due_date < today and a.status != "compliant")
    due_soon_ads = sum(1 for a in all_ads if a.due_date and today <= a.due_date <= in30 and a.status not in ("compliant","n/a"))
    compliant    = sum(1 for a in all_ads if a.status == "compliant")
    overdue_chks = sum(1 for c in all_chks if c.next_due_date and c.next_due_date < today and c.status != "completed")
    due_soon_chks= sum(1 for c in all_chks if c.next_due_date and today <= c.next_due_date <= in30)

    llava_docs  = sum(1 for d in all_docs if d.structured_data and d.structured_data.get("_source") == "llava")
    regex_docs  = sum(1 for d in all_docs if d.structured_data and d.structured_data.get("_source") in ("regex","regex_fallback"))
    ollama_docs = sum(1 for d in all_docs if d.structured_data and d.structured_data.get("_source") == "ollama")

    ata_counts:  dict = {}
    status_counts: dict = {}
    type_counts:  dict = {}
    for d in all_docs:
        sd = d.structured_data or {}
        for code in sd.get("ata_codes", []):
            ata_counts[code] = ata_counts.get(code, 0) + 1
        ms = sd.get("maintenance_status", "Unknown")
        status_counts[ms] = status_counts.get(ms, 0) + 1
        dt = d.document_type or "general"
        type_counts[dt] = type_counts.get(dt, 0) + 1

    return {
        "total_aircraft":        db.query(Aircraft).count(),
        "total_documents":       len(all_docs),
        "overdue_ads":           overdue_ads,
        "due_soon_ads":          due_soon_ads,
        "compliant_ads":         compliant,
        "overdue_checks":        overdue_chks,
        "due_soon_checks":       due_soon_chks,
        "ai_extracted_docs":     llava_docs + ollama_docs,
        "llava_extracted_docs":  llava_docs,
        "regex_extracted_docs":  regex_docs,
        "ata_distribution":      dict(sorted(ata_counts.items(), key=lambda x: -x[1])[:10]),
        "status_distribution":   status_counts,
        "doc_type_distribution": type_counts,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ALERTS
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/alerts", tags=["Stats"])
def get_alerts(db: Session = Depends(get_db)):
    today = date.today().isoformat()
    in30  = date.today().replace(day=min(date.today().day+30,28)).isoformat()
    alerts = []
    for a in db.query(AirworthinessDirective).all():
        ac = db.query(Aircraft).filter(Aircraft.id == a.aircraft_id).first()
        reg = ac.registration if ac else str(a.aircraft_id)
        if a.due_date and a.due_date < today and a.status != "compliant":
            alerts.append({"severity":"critical","type":"AD Overdue","reference":a.ad_number,
                           "aircraft":reg,"due_date":a.due_date,"description":a.description or ""})
        elif a.due_date and today <= a.due_date <= in30 and a.status not in ("compliant","n/a"):
            alerts.append({"severity":"warning","type":"AD Due Soon","reference":a.ad_number,
                           "aircraft":reg,"due_date":a.due_date,"description":a.description or ""})
    for c in db.query(MaintenanceCheck).all():
        ac = db.query(Aircraft).filter(Aircraft.id == c.aircraft_id).first()
        reg = ac.registration if ac else str(c.aircraft_id)
        if c.next_due_date and c.next_due_date < today and c.status != "completed":
            alerts.append({"severity":"critical","type":f"{c.check_type}-Check Overdue",
                           "reference":c.work_order or "","aircraft":reg,
                           "due_date":c.next_due_date,"description":c.notes or ""})
        elif c.next_due_date and today <= c.next_due_date <= in30:
            alerts.append({"severity":"warning","type":f"{c.check_type}-Check Due Soon",
                           "reference":c.work_order or "","aircraft":reg,
                           "due_date":c.next_due_date,"description":c.notes or ""})
    alerts.sort(key=lambda x: (0 if x["severity"]=="critical" else 1, x["due_date"] or ""))
    return {"count": len(alerts), "alerts": alerts}

# ─────────────────────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────────────────────

@app.post("/signup", response_model=UserResponse)
def signup(user: UserSignup, db: Session = Depends(get_db)):

    existing_user = (
        db.query(User)
        .filter(User.email == user.email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

   new_user = User(
    full_name=user.name,
    email=user.email,
    password_hash=hash_password(user.password)
)

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):

    existing_user = (
        db.query(User)
        .filter(User.email == user.email)
        .first()
    )

    if not existing_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    if not verify_password(
        user.password,
        existing_user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    return {
    "message": "Login successful",
    "id": existing_user.id,
    "name": existing_user.full_name,
    "email": existing_user.email
}