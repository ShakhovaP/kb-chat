#routes.py
from fastapi import FastAPI, UploadFile, HTTPException, Query
import uvicorn
import tempfile
import os
import logging
from services.knowledge_base_service import KnowledgeBaseService
from services.chat_service import ChatService
from services.analysis_service import AnalysisService
from pydantic import BaseModel
from typing import Dict, List, Optional
import uuid
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Processing Service")
kb = KnowledgeBaseService()
chat = ChatService()
analysis = AnalysisService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or set specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SpeechLanguage(str, Enum):
    DANISH = "da-DK"
    ENGLISH_US = "en-US"
    
class ConversationRequest(BaseModel):
    query: str
    session_id: str
    top_k: int = 5

chat_histories: Dict[str, List[dict]] = {}
def get_chat_history(session_id: str = Query(None)):
    if not session_id:
        # Generate a new session ID if none provided
        session_id = str(uuid.uuid4())
    
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    return session_id, chat_histories[session_id]


@app.post("/kb_upload")
async def process_file(
    file: UploadFile, 
    url: str,
    speech_language: Optional[SpeechLanguage] = Query(
        default=None,
        description="BCP-47 language code for speech recognition"
    )):
    """
    Process PDF, PPTX, or MP4 files: extract content, structure it, generate embeddings, and store in search.
    """
    # Validate file extension
    if not file.filename.endswith(('.pdf', '.pptx', '.mp4')):
        return {"error": "Unsupported file format. Please upload PDF, PPTX, or MP4 files only."}
    
    # Get file extension to determine processing method
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # Create temporary file with appropriate extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    try:
        if file_extension == '.pdf':
            result = await kb.process_pdf_with_link(temp_file_path, file.filename, url)
            file_type = "PDF"
        elif file_extension == '.pptx':
            result = await kb.process_pptx_with_link(temp_file_path, file.filename, url)
            file_type = "PPTX"
        elif file_extension == '.mp4':
            result = await kb.process_video_with_link(temp_file_path, file.filename, url, speech_language)
            file_type = "video"
        
        return {
            "message": f"{file_type} processed successfully",
            "filename": file.filename,
            "document_id": result.get("document_id")
        }
    
    except Exception as e:
        logger.error(f"Error processing {file_extension} file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.get("/kb_search")
async def search_knowledge_base(
    query: str = Query(..., description="Search query text"),
    top_k: int = Query(5, description="Number of results to return"),
    # document_id: str = Query(None, description="Optional: Limit search to specific document")
):
    """
    Search the knowledge base for relevant information based on the provided query.
    """
    try:
        results = await kb.search(
            query=query,
            top_k=top_k,
        )
        
        return {
            "results": results,
            "query": query,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching knowledge base: {str(e)}")
    
@app.post("/excel_upload")
async def process_excel(file: UploadFile, session_id: str = Query(...)):
    """
    Process an Excel file for NPS analysis
    """
    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files are supported")
    
    # Create temporary file to store uploaded Excel
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    try:
        analysis_results = analysis.analyze_document(temp_file_path, session_id)        
        return analysis_results
    
    except Exception as e:
        logger.error(f"Error processing Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing Excel: {str(e)}")
    
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

@app.post("/chat")
async def answer_question(
    query: str = Query(..., description="Search query text"),
    # top_k: int = Query(5, description="Number of knowledge base results to consider"),
    # chat_history: list[str] = Query([], description="Previous messages in the conversation")  # New parameter
    sessionId: str = Query(..., description="Previous messages in the conversation")  # New parameter
):
    """
    Answer user questions using knowledge base search and Azure OpenAI.
    """
    logger.info("/chat")
    # print("/chat")
    try:
        results = await chat.chat(query=query, session_id=sessionId)
        return results

    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)