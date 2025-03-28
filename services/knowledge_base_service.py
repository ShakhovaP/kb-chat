import os
import time
import logging
from typing import List, Dict, Optional
from io import BytesIO

import tiktoken
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter

from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, 
    VectorSearch, VectorSearchAlgorithmConfiguration, 
    VectorSearchProfile, HnswParameters
)
from azure.search.documents.models import VectorizedQuery

from langchain_openai import AzureChatOpenAI
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
    # create_stuff_documents_chain,
    # ConversationChain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.retrievers import AzureAISearchRetriever
from services.config_manager import ConfigManager
from services.azure_service import AzureServiceClients

# Configure logging and environment variables
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class KnowledgeBaseService:
    """
    Comprehensive service for processing, vectorizing, and searching documents.
    """
    def __init__(self):
        """
        Initialize the Knowledge Base Service with Azure clients and configurations.
        """
        self.config_manager = ConfigManager()
        self.azure_services = AzureServiceClients(self.config_manager)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration parameters
        self.index_name = self.config_manager.get_required_env("AZURE_SEARCH_INDEX_NAME")
        self.openai_deployment = self.config_manager.get_required_env("AZURE_OPENAI_GPT_DEPLOYMENT")
        self.openai_embeddings_deployment = self.config_manager.get_required_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.mongodb_uri = self.config_manager.get_required_env("MONGODB_URI")
        self.embedding_dimensions = 1536

        # Initialize retriever and LLM
        self.retriever = AzureAISearchRetriever(
            content_key="content", 
            top_k=1, 
            index_name=self.index_name
        )

        self.llm = AzureChatOpenAI(
            azure_deployment=self.openai_deployment,
            api_version=self.config_manager.get_required_env("AZURE_OPENAI_API_VERSION"),
            temperature=0,
            max_tokens=None,
            verbose=True
        )

    def extract_text_from_pdf(self, file_path: str) -> List[str]:
        """
        Extracts text from a PDF file using Azure Form Recognizer.
        
        :param file_path: Path to the PDF file
        :return: List of extracted text, split by 2 pages
        """
        try:
            input_pdf = PdfReader(file_path)
            batch_size = 2 # if len(input_pdf.pages) > 1 else 1
            text_extracted = []
            
            for page in range(0, len(input_pdf.pages), batch_size):
                batch_pdf = PdfWriter()
                end_page = min(page + batch_size, len(input_pdf.pages))
                
                for i in range(page, end_page):
                    batch_pdf.add_page(input_pdf.pages[i])

                with BytesIO() as batch_output:
                    batch_pdf.write(batch_output)
                    batch_output.seek(0)
                    
                    poller = self.azure_services.document_analysis_client.begin_analyze_document(
                        "prebuilt-layout", batch_output
                    )
                    result = poller.result()
                    
                text_extracted.append(result.content)
                
            return text_extracted
        
        except Exception as e:
            self.logger.error(f"PDF text extraction error: {e}")
            raise

    def structure_content_with_openai(self, text: str) -> str:
        """
        Structure extracted text using Azure OpenAI.
        
        :param text: Raw text extracted from PDF
        :return: Structured content
        """
        system_prompt = """
        You are an advanced AI system specialized in structuring OCR-extracted text. 
        Tasks:
        - Correct OCR misspellings
        - Remove redundant whitespace, page numbers, headers, footers, and colon titles 
        (e.g., repeated author names, chapter titles, or book/publication names that appear 
        at the top or bottom of the page)
        - Eliminate parsed text from images, such as book titles and author information
        - Preserve paragraph breaks and section structure
        - Do not modify or alter the meaning of the content
        - Format the output so that each paragraph is separated by two line breaks (double enter)
        - Return only the cleaned and structured text, without any additional explanations or metadata
        """
        try:
            response = self.azure_services.azure_openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            self.logger.error(f"Text structuring error: {e}")
            return text

    async def process_pdf_file(self, file_path: str) -> Dict[str, str]:
        """
        Process a PDF file through the entire pipeline.
        
        :param file_path: Path to the PDF file
        :return: Processing result with document ID
        """
        try:
            text_arr = self.extract_text_from_pdf(file_path)
            file_name = os.path.basename(file_path).replace(".pdf", "")

            structured_text_arr = []
            chunks = []
            
            for i, text in enumerate(text_arr):
                structured_data = self.structure_content_with_openai(text)
                structured_text_arr.append(structured_data)

                split_chunks = self.split_text_into_chunks(structured_data)
                
                chunks.extend([
                    {
                        "id": f"{file_name}_doc{i}_c{j}",
                        "content": chunk,
                        "embedding": self.generate_embeddings(chunk)
                    }
                    for j, chunk in enumerate(split_chunks)
                ])

            self.upload_vectors_to_search(chunks)
            
            return {
                "document": file_path,
                "status": "processed"
            }
        
        except Exception as e:
            self.logger.error(f"PDF processing error: {str(e)}")
            raise

    def split_text_into_chunks(self, text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks respecting paragraph boundaries.
        
        :param text: Input text
        :param max_tokens: Maximum tokens per chunk
        :param overlap: Number of tokens for chunk overlap
        :return: List of text chunks
        """
        encoding = tiktoken.encoding_for_model(self.openai_deployment)
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        current_chunk_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = len(encoding.encode(paragraph))
            
            # If a single paragraph exceeds max_tokens, split it by sentences
            if paragraph_tokens > max_tokens:
                sentences = paragraph.replace('. ', '.\n').split('\n')
                # sentences = paragraph.split('.')
                for sentence in sentences:
                    sentence_tokens = len(encoding.encode(sentence))
                    
                    # If adding this sentence would exceed max_tokens, start a new chunk
                    if current_chunk_tokens + sentence_tokens > max_tokens:
                        chunks.append(current_chunk)
                        # Keep some overlap by keeping the last sentence if possible
                        overlap_text = ""
                        if current_chunk:
                            last_sentence = current_chunk.split('. ')[-1]
                            if len(encoding.encode(last_sentence)) < overlap:
                                overlap_text = last_sentence + '. '
                        current_chunk = overlap_text + sentence
                        current_chunk_tokens = len(encoding.encode(current_chunk))
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                        current_chunk_tokens += sentence_tokens
            else:
                # If adding this paragraph would exceed max_tokens, start a new chunk
                if current_chunk_tokens + paragraph_tokens > max_tokens:
                    chunks.append(current_chunk)
                    current_chunk = paragraph
                    current_chunk_tokens = paragraph_tokens
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                    current_chunk_tokens += paragraph_tokens
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
        
    def generate_embeddings(self, text):
        """
        Generates embeddings using Azure OpenAI Embedding model.
        
        Args:
            text (str): Text to generate embeddings for
            
        Returns:
            list: Vector embeddings
        """
        self.logger.info("Generating embeddings")
        response = self.azure_services.azure_openai_client.embeddings.create(
            input=text,
            model=self.openai_embeddings_deployment,
            dimensions=self.embedding_dimensions
        )
        vector = response.data[0].embedding
        return vector
    
    def create_search_index(self):
        """Create Azure AI Search index"""
        # logger.info("Checking if search index exists")
        
        # Check if index exists
        idx_exists = False
        self.logger.info(f"list_indexes: {self.azure_services.search_index_client.list_indexes()}")
        if self.index_name in [index.name for index in self.azure_services.search_index_client.list_indexes()]:
            self.logger.info(f"Index {self.index_name} already exists")
            idx_exists = True
        
        # Create index with vector search capability
        fields = [
            SearchField(
                name="id", 
                type=SearchFieldDataType.String, 
                key=True,
                sortable=True,
                filterable=True,
            ),
            SearchField(name="content", type=SearchFieldDataType.String),
            # SearchField(name="metadata", type=SearchFieldDataType.String, searchable=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.embedding_dimensions,
                vector_search_configuration="vector-config",  # This should match the profile name
            )
        ]       
        vector_search = VectorSearch(
            algorithms={
                "hnsw": VectorSearchAlgorithmConfiguration(
                    name="hnsw",
                    kind="hnsw",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric="cosine"
                    )
                )
            },
            profiles={
                "vector-config": VectorSearchProfile(  # Change this name to match what you use in SearchField
                    name="vector-config",  # This should match
                    algorithm_configuration_name="hnsw"
                )
            }
        )
        
        index = SearchIndex(
            name=self.index_name, 
            fields=fields, 
            vector_search=vector_search,
            # semantic_search=semantic_search,
        )
        self.logger.info(f"{'Updating' if idx_exists else 'Creating'}  index {self.index_name}")
        self.azure_services.search_index_client.create_index(index)

    def upload_vectors_to_search(self, documents):
    # def upload_vectors_to_search(embeddings):
        """
        Upload the generated embeddings to Azure AI Search
        """        
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            try:
                results = self.azure_services.search_client.upload_documents(documents=batch)
                success_count = sum(1 for r in results if r.succeeded)
                self.logger.info(f"Indexed {success_count}/{len(batch)} documents.")
            except Exception as e:
                self.logger.exception(f"Error uploading batch to search index: {e}")
            
            time.sleep(1)

    async def chat(self, query: str, session_id: str = "anonymous") -> Dict[str, str]:
        """
        Chat method with context-aware retrieval and response generation.
        
        :param query: User's chat query
        :param session_id: Unique session identifier
        :return: Chat response dictionary
        """
        try:
            message_history = MongoDBChatMessageHistory(
                connection_string=self.mongodb_uri, 
                session_id=session_id
            )

            contextualize_q_system_prompt = (
                """
                Given a chat history and the latest user question
                which might reference context in the chat history,
                formulate a standalone question which can be understood
                without the chat history. Do NOT answer the question, just
                reformulate it if needed and otherwise return it as is.
                """
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retriever, contextualize_q_prompt
            )

            # Answer question
            qa_system_prompt = (
                '''You are a helpful assistant that answers questions based on the provided pieces of retrieved context. 
                If the context doesn't contain relevant information to answer the question,
                state that you don't have enough information and provide a general response based on your knowledge.
                {context}'''

            ) 
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                ("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"),

                ]
            )  
            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt) 
            rag_chain = create_retrieval_chain(
                history_aware_retriever, question_answer_chain
            )
            self.logger.info("before responce")

            response = rag_chain.invoke({"input": query, "chat_history": message_history.messages})

            self.logger.info("response", response)
            message_history.add_user_message(query)
            message_history.add_ai_message(response['answer'])
            return {
                "question": query,
                "answer": response['answer'],
                # "source": "knowledge_base" if kb_results else "model_general_knowledge",
                # "kb_results_count": len(kb_results) if kb_results else None,
            }
        
        except Exception as e:
            self.logger.error(f"Chat method error: {e}")
            raise

def main():
    """
    Application entry point for knowledge base service.
    """
    kb_service = KnowledgeBaseService()
    # Additional setup or initialization can be added here

if __name__ == "__main__":
    main()