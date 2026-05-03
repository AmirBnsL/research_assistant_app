import uuid
import asyncio
import os
import shutil
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    BackgroundTasks,
    status,
    HTTPException,
    Form,
)
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from src.api.database import get_db, Document
from src.rag.ingestion import extract_text_from_pdf, chunk_academic_paper
from src.rag.embedding import embed_and_store_batch

router = APIRouter(prefix="/api/documents", tags=["documents"])

job_progress = {}


def process_document_background(
    doc_id: str,
    file_path: str,
    filename: str,
    db: Session,
    session_id: str | None = None,
):
    try:
        db_doc = db.query(Document).filter(Document.id == doc_id).first()
        if db_doc:
            db_doc.status = "processing"
            db.commit()

        job_progress[doc_id] = 5

        raw_text = extract_text_from_pdf(file_path)
        job_progress[doc_id] = 15

        chunks = chunk_academic_paper(raw_text)
        job_progress[doc_id] = 20

        embed_and_store_batch(
            chunks,
            filename,
            doc_id,
            job_progress,
            start_progress=20,
            session_id=session_id,
        )

        if db_doc:
            db_doc.status = "completed"
            db.commit()

    except Exception as e:
        if db_doc:
            db_doc.status = f"error: {str(e)}"
            db.commit()
        job_progress[doc_id] = -1


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    session_id: str = Form(None),
):
    doc_id = str(uuid.uuid4())
    upload_dir = "./data/user_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{doc_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    db_doc = Document(
        id=doc_id,
        filename=file.filename or "<unknown>",
        status="uploading",
        session_id=session_id,
    )
    db.add(db_doc)
    db.commit()

    background_tasks.add_task(
        process_document_background,
        doc_id,
        file_path,
        file.filename or "<unknown>",
        db,
        session_id,
    )
    return {"doc_id": doc_id, "status": "Processing started", "session_id": session_id}


@router.get("/progress/{doc_id}")
async def get_progress(doc_id: str):
    async def event_generator():
        last_reported_progress = -1
        while True:
            current_status = job_progress.get(doc_id, 0)
            if current_status != last_reported_progress:
                yield {
                    "event": "message",
                    "data": f'{{ "progress": {current_status}, "status": "processing" }}',
                }
                last_reported_progress = current_status

            if current_status >= 100:
                yield {
                    "event": "complete",
                    "data": '{ "progress": 100, "status": "completed" }',
                }
                break
            if current_status == -1:
                yield {"event": "error", "data": '{ "status": "failed" }'}
                break
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@router.get("/", response_model=list[dict])
async def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).all()
    return [
        {"id": doc.id, "filename": doc.filename, "status": doc.status} for doc in docs
    ]


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str, db: Session = Depends(get_db)):
    db_doc = db.query(Document).filter(Document.id == doc_id).first()
    if not db_doc:
        raise HTTPException(status_code=404, detail="Document not found")

    upload_dir = "./data/user_uploads"
    file_path = os.path.join(upload_dir, f"{doc_id}_{db_doc.filename}")
    if os.path.exists(file_path):
        os.remove(file_path)

    if doc_id in job_progress:
        del job_progress[doc_id]

    db.delete(db_doc)
    db.commit()
    return None
