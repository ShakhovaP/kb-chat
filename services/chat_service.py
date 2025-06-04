import logging
from typing import Dict
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import (
    # create_history_aware_retriever,
    create_retrieval_chain
    # history_aware_retriever,
    # retrieval_chain
)
# from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import AzureAISearchRetriever
from services.config_manager import ConfigManager
from services.azure_service import AzureServiceClients
from langchain_community.vectorstores import AzureSearch


# Configure logging and environment variables
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ChatService:
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
        self.mongodb_uri = self.config_manager.get_required_env("MONGODB_URI")

        # Initialize retriever and LLM
        self.retriever = AzureAISearchRetriever(
            content_key="content", 
            top_k=7, 
            index_name=self.index_name
        )

        self.llm = AzureChatOpenAI(
            azure_deployment=self.openai_deployment,
            api_version=self.config_manager.get_required_env("AZURE_OPENAI_API_VERSION"),
            temperature=0,
            max_tokens=None,
            verbose=True
        )

    async def chat(self, query: str, session_id: str = "anonymous") -> Dict[str, str]:
        """
        Chat method with context-aware retrieval and response generation.
        
        :param query: User's chat query
        :param session_id: Unique session identifier
        :return: Chat response dictionary
        """
        try:
            print('\n\nchat() start')
            message_history = MongoDBChatMessageHistory(
                connection_string=self.mongodb_uri, 
                session_id=session_id
            )

            contextualize_q_system_prompt = (
                """
                Given a chat history and the latest user question which might reference context in the 
                chat history, formulate a standalone question which can be understood without the chat history.
                Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
                To the generated question add the request to search for Danish companies examples and their methods
                related to the topic.
                """
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}. "),
                ]
            )
            history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retriever, contextualize_q_prompt
            )
            retrieved_documents = history_aware_retriever.invoke({
                "input": query,
                "chat_history": message_history.messages
            })

            document_info = []
            for doc in retrieved_documents:
                doc_id = doc.metadata.get('id', 'unknown_id')
                doc_url = doc.metadata.get('file_url', '')
                score = doc.metadata.get('@search.score', 'N/A')  # Get score from metadata if available
                # print("score", doc.metadata.get('@search.score', 'N/A'))
                # if float(score)>0: document_info.append((doc_url, score))
                if float(score)>0: document_info.append(doc_url)
            # references = "\nReferences:" if len(document_info)>0 else ""
            # for i, (doc_url, score) in enumerate(document_info[:3]):
            #     # references += f"\n{i}. {doc_url} (relevance score: {score})"
            #     references += f"\n{i+1}. {doc_url}"
            # Answer question
            qa_system_prompt = (
                """You are a helpful assistant that answers questions based on the provided pieces of retrieved context. When responding:
                1. Present the answer in two parts:
                - **Results**: A short text summary.
                - **Key Takeaways**: 3 concise bullet points highlighting the most important insights.
                2. If possible, highlight specific examples of Danish companies mentioned in the retrieved documents.
                - Do NOT mention if such examples are missing.
                3. If the context doesn't contain relevant information, provide a general response based on your knowledge.
                - Do NOT inform the user that the context was insufficient.
                4. End your response with a relevant follow-up question.
                {context}"""

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

            response = rag_chain.invoke({"input": query, "chat_history": message_history.messages})
            self.logger.info("response", response)
            message_history.add_user_message(query)
            # message_history.add_ai_message(response['answer'] + references)
            message_history.add_ai_message(response['answer'])
            return {
                "question": query,
                # "answer": response['answer'] + references
                "sources": set(document_info),
                "answer": response['answer']
            }
        
        except Exception as e:
            self.logger.error(f"Chat method error: {e}")
            raise
