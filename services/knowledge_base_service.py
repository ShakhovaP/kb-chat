import os
import time
import logging
from typing import List, Dict
from io import BytesIO
from azure.search.documents.models import VectorizedQuery
import tiktoken
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter

from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, 
    VectorSearch, SearchableField, SimpleField,
    VectorSearchProfile, HnswAlgorithmConfiguration, 
)
from services.config_manager import ConfigManager
from services.azure_service import AzureServiceClients
import tempfile
import moviepy as mp
import azure.cognitiveservices.speech as speechsdk
from spire.presentation import *
from spire.presentation.common import *

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
        self.embedding_dimensions = 1536

    async def search(self, query, top_k=5):
        """
        Search the knowledge base for relevant content.
        
        Args:
            query_text (str): The search query
            top_k (int): Number of results to return
            document_id (str, optional): Limit search to specific document
            
        Returns:
            list: Search results
        """
        emb = self.generate_embeddings(query)
        vector_query = VectorizedQuery(vector=emb, k_nearest_neighbors=top_k, fields="embedding")

        results = self.azure_services.search_client.search(
            vector_queries=[vector_query],
            select=['id', 'content'],
            include_total_count=True,
        )
        
        query_results = []
        for r in results:
            query_results.append({
                'content': r['content'],
                'score': r['@search.score'] 
            })
            print(r['content'])

            
        return query_results

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
    
    def extract_audio(self, video_path):
        # Create a temporary file with .wav extension
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_path = temp_audio_file.name
        
        try:
            print(f"Extracting audio from {video_path}")
            video = mp.VideoFileClip(video_path)
            video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
            
            return temp_audio_path
        except Exception as e:
            try:
                os.unlink(temp_audio_path)
            except:
                pass
            raise e

    def process_speech(self, video_path, speech_config):
        temp_audio_path = None
        try:
            temp_audio_path = self.extract_audio(video_path=video_path)
            
            audio_config = speechsdk.AudioConfig(filename=temp_audio_path)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, 
                audio_config=audio_config
            )
            
            # For collecting the complete transcription
            all_results = []
            
            # Set up event handlers for continuous recognition
            done = False
            def handle_final_result(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    all_results.append(evt.result.text)
            
            def stop_callback(evt):
                nonlocal done
                done = True
            
            # Connect the event handlers
            speech_recognizer.recognized.connect(handle_final_result)
            speech_recognizer.session_stopped.connect(stop_callback)
            speech_recognizer.canceled.connect(stop_callback)
            
            # Start continuous recognition
            speech_recognizer.start_continuous_recognition()
            
            # Wait for recognition to complete
            while not done:
                time.sleep(0.5)
                
            # Stop recognition
            speech_recognizer.stop_continuous_recognition()
            
            # Combine all results
            complete_transcription = ' '.join(all_results)
            
            return complete_transcription
        finally:
            # Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    # Make sure to delete all references that might keep the file open
                    speech_recognizer = None
                    audio_config = None
                    
                    # Try to delete, but don't crash if it fails
                    os.unlink(temp_audio_path)
                    print(f"Temporary audio file deleted")
                except PermissionError:
                    print(f"Could not delete temporary file: {temp_audio_path}")

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
                "document": file_name,
                "status": "processed"
            }
        
        except Exception as e:
            self.logger.error(f"PDF processing error: {str(e)}")
            raise
    
    async def process_pdf_with_link(self, file_path: str, file_name: str, url: str) -> Dict[str, str]:
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
                        "embedding": self.generate_embeddings(chunk),
                        "file_name":file_name,
                        "file_url": url,
                    }
                    for j, chunk in enumerate(split_chunks)
                ])

            self.upload_vectors_to_search(chunks)
            
            return {
                "document": file_name,
                "status": "processed"
            }
        
        except Exception as e:
            self.logger.error(f"PDF processing error: {str(e)}")
            raise
    def get_transcription(self, file_path: str, speech_language: str | None) -> str:
        speech_config = self.azure_services.speech_config
        if speech_language: speech_config.speech_recognition_language=speech_language
        transcription = self.process_speech(file_path, speech_config)
        return transcription

    async def process_video_with_link(self, file_path: str, file_name: str, url: str, speech_language: str | None) -> Dict[str, str]:
        """
        Process a video file through the entire pipeline.
        
        :param file_path: Path to the video file
        :return: Processing result with document ID
        """
        try:
            transcription = self.get_transcription(file_path, speech_language)

            file_name = os.path.basename(file_path).replace(".mp4", "")

            chunks = []
            split_chunks = self.split_text_into_chunks(transcription)
                
            chunks.extend([
                {
                    "id": f"{file_name}_c{j}",
                    "content": chunk,
                    "embedding": self.generate_embeddings(chunk),
                    "file_name":file_name,
                    "file_url": url,
                }
                for j, chunk in enumerate(split_chunks)
            ])

            self.upload_vectors_to_search(chunks)
            
            return {
                "document": file_name,
                "status": "processed",
                "transcription": transcription
            }
        
        except Exception as e:
            self.logger.error(f"Video processing error: {str(e)}")
            raise

    async def process_pptx_with_link(self, file_path: str, file_name: str, url: str) -> Dict[str, str]:
        try:
            presentation = Presentation()
            presentation.LoadFromFile(file_path)
            # Or load a PowerPoint presentation in PPT format
            #presentation.LoadFromFile("Sample.ppt")

            # Convert the presentation to PDF format
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf_file:
                temp_pdf_path = temp_pdf_file.name
            presentation.SaveToFile(temp_pdf_path, FileFormat.PDF)
            presentation.Dispose()

            result = await self.process_pdf_with_link(temp_pdf_path, file_name, url)
            return result
        finally:
            # Clean up temporary file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    # Try to delete, but don't crash if it fails
                    os.unlink(temp_pdf_path)
                    print(f"Temporary pdf file deleted")
                except PermissionError:
                    print(f"Could not delete temporary file: {temp_pdf_path}")

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
        idx_exists = False
        print(f"list_indexes: {self.azure_services.search_index_client.list_indexes()}")
        if self.index_name in [index.name for index in self.azure_services.search_index_client.list_indexes()]:
            print(f"Index {self.index_name} already exists")
            idx_exists = True

        # Create index with vector search capability
        fields = [
            SimpleField(
                name="id", 
                type=SearchFieldDataType.String, 
                key=True,
                sortable=True,
                filterable=True,
            ),
            SearchableField(name="file_name", type=SearchFieldDataType.String, searchable=True),
            SearchableField(name="file_url", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=self.embedding_dimensions, 
                vector_search_profile_name="vector-config")
            
        ]       
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(
                  name="vector-config",
                  algorithm_configuration_name="hnsw"
            )]
        )


        index = SearchIndex(
            name=self.index_name, 
            fields=fields, 
            vector_search=vector_search,
            # semantic_search=semantic_search,
        )
        print(f"{'Updating' if idx_exists else 'Creating'}  index {self.index_name}")
        self.azure_services.search_index_client.create_or_update_index(index)
        
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
