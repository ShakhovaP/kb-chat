from services.config_manager import ConfigManager
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk

class AzureServiceClients:
    """
    Centralized management of Azure service clients.
    """
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize Azure service clients.
        
        :param config_manager: Configuration management instance
        """
        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=config_manager.get_required_env("AZURE_FORM_RECOGNIZER_ENDPOINT"),
            credential=AzureKeyCredential(config_manager.get_required_env("AZURE_FORM_RECOGNIZER_KEY"))
        )
        
        self.azure_openai_client = AzureOpenAI(
            azure_endpoint=config_manager.get_required_env("AZURE_OPENAI_ENDPOINT"),
            api_key=config_manager.get_required_env("AZURE_OPENAI_API_KEY"),
            api_version=config_manager.get_required_env("AZURE_OPENAI_API_VERSION")
        )
        
        self.search_index_client = SearchIndexClient(
            endpoint=config_manager.get_required_env("AZURE_SEARCH_ENDPOINT"),
            credential=AzureKeyCredential(config_manager.get_required_env("AZURE_AI_SEARCH_API_KEY"))
        )
        
        self.search_client = SearchClient(
            endpoint=config_manager.get_required_env("AZURE_SEARCH_ENDPOINT"),
            index_name=config_manager.get_required_env("AZURE_SEARCH_INDEX_NAME"),
            credential=AzureKeyCredential(config_manager.get_required_env("AZURE_AI_SEARCH_API_KEY"))
        )
        self.speech_config = speechsdk.SpeechConfig(subscription=config_manager.get_required_env("AZURE_SPEECH_KEY"), region=config_manager.get_required_env("AZURE_SPEECH_REGION"))