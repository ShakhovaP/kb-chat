from fastapi import FastAPI, UploadFile, HTTPException, Query
import uvicorn
import tempfile
import os
import logging
from services.knowledge_base_service import KnowledgeBaseService
from pydantic import BaseModel
from typing import Dict, List, Optional
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Processing Service")
kb = KnowledgeBaseService()

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
async def process_pdf(file: UploadFile):
    """
    Process a PDF file: extract text, structure content, generate embeddings, and store in search.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create temporary file to store uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    try:
        result = await kb.process_pdf_file(temp_file_path)
        return {
            "message": "PDF processed successfully",
            "filename": file.filename,
            "document_id": result.get("document_id")
        }
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    finally:
        # Clean up temporary file
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
    print("/chat")
    try:
        results = await kb.chat(query=query, session_id=sessionId)
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