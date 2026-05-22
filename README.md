# AeroDoc — Aircraft Technical Records Platform

FlyDocs-style aviation records management platform with Claude AI extraction.

---

## Project Structure

```
aerodoc/
│
├── app/
│   ├── __init__.py
│   ├── db.py          ← Database connection (PostgreSQL)
│   ├── models.py      ← SQLAlchemy ORM models
│   ├── schemas.py     ← Pydantic request/response schemas
│   └── main.py        ← FastAPI routes (all endpoints)
│
├── index.html         ← Frontend (open in browser)
├── requirements.txt   ← Python dependencies
├── .env.example       ← Copy to .env and fill in
├── start_all.bat      ← One-click startup (Windows)
└── README.md
```

---

## Setup Instructions (Windows)

### Step 1 — Create the PostgreSQL database

Open pgAdmin or psql and run:

```sql
CREATE DATABASE aerodoc;
```

### Step 2 — Create your .env file

Copy `.env.example` to `.env` in the same folder:

```
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/aerodoc
ANTHROPIC_API_KEY=sk-ant-XXXXXXXXXXXXXXXX
```

Replace `yourpassword` with your actual PostgreSQL password.
The `ANTHROPIC_API_KEY` is optional — without it, regex extraction is used.

### Step 3 — Create Python virtual environment

Open a terminal in the `aerodoc` folder:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4 — Start the backend

Double-click `start_all.bat`, or run manually:

```bat
venv\Scripts\activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API will start at: http://127.0.0.1:8000
API docs (Swagger): http://127.0.0.1:8000/docs

### Step 5 — Open the frontend

Open `index.html` directly in your browser, or serve it via VS Code Live Server.

> **Note**: The frontend connects to `http://127.0.0.1:8000` by default.
> Make sure the backend is running before opening the frontend.

---

## Features

### Fleet Registry
- Register aircraft with MSN, type, operator, manufacture date, FH/FC
- Status tracking: Active / Grounded / AOG / Stored

### Document Management
- Upload PDF documents per aircraft
- Claude AI auto-extraction: registration, ATA codes, part numbers, serial numbers, AME names, dates, AD/SB references, defect descriptions, maintenance status
- Fallback regex extraction if Claude API key is not set
- Full-text search across all extracted text
- Filter by ATA chapter, document type, aircraft, maintenance status, personnel

### Airworthiness Directives (ADs)
- Track AD compliance per aircraft
- Status: Open / Compliant / Overdue / N/A
- Due date tracking with automatic alerts

### Service Bulletins (SBs)
- Track SB compliance per aircraft
- Link compliance documents to SB records

### Maintenance Checks
- A / B / C / D check scheduling
- Next due date and flight hours tracking
- Progress visualization

### Alerts
- Real-time critical and warning alerts on dashboard
- Overdue ADs, expiring CRS, upcoming maintenance checks

### Analytics
- ATA chapter distribution
- Maintenance status breakdown
- Document type distribution

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /aircraft | List all aircraft |
| POST | /aircraft | Register aircraft |
| PUT | /aircraft/{id} | Update aircraft |
| DELETE | /aircraft/{id} | Delete aircraft |
| GET | /documents | List all documents |
| POST | /documents/upload | Upload + extract PDF |
| GET | /documents/search?query= | Full-text search |
| GET | /documents/filter/by-aircraft?aircraft_id= | Filter by aircraft |
| GET | /documents/filter/by-ata?ata= | Filter by ATA code |
| GET | /documents/filter/by-chapter?chapter= | Filter by chapter |
| GET | /documents/filter/by-status?status= | Filter by status |
| POST | /documents/{id}/reanalyse | Re-run Claude AI on existing doc |
| GET | /documents/{id}/download | Download file |
| GET | /ads | List airworthiness directives |
| POST | /ads | Add AD |
| PUT | /ads/{id} | Update AD |
| DELETE | /ads/{id} | Delete AD |
| GET | /sbs | List service bulletins |
| POST | /sbs | Add SB |
| DELETE | /sbs/{id} | Delete SB |
| GET | /checks | List maintenance checks |
| POST | /checks | Add maintenance check |
| DELETE | /checks/{id} | Delete check |
| GET | /alerts | Get all active alerts |
| GET | /stats/summary | Fleet summary statistics |

---

## Troubleshooting

**"DATABASE_URL is not set"**
→ Make sure `.env` file exists in the same folder as `start_all.bat`

**"Cannot reach API at http://127.0.0.1:8000"**
→ Backend is not running. Run `start_all.bat` first.

**CORS errors in browser**
→ The API allows all origins. Make sure you're hitting port 8000.

**Claude AI extraction not working**
→ Check `ANTHROPIC_API_KEY` in `.env`. Platform falls back to regex automatically.

**PostgreSQL won't start**
→ Check your PostgreSQL service name. Edit `start_all.bat` line:
   `net start postgresql-x64-16` → replace `16` with your version number.
