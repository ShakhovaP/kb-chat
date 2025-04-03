import os
import chainlit as cl
from chainlit.input_widget import TextInput, Slider
import aiohttp
import json
from bson import ObjectId

# Configuration
# API_BASE_URL = "http://localhost:8000"
API_BASE_URL = "http://fastapi-backend:8000"


@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    session_id = str(ObjectId())
    cl.user_session.set("session_id", session_id)
    welcome_msg = await cl.Message(
        content="Welcome to the Knowledge Base Assistant! You can upload PDFs or search the knowledge base."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages and maintain chat history."""
    user_message = message.content.strip()

    # Create a loading message that will be displayed while waiting for the response
    msg = cl.Message(content="Searching the knowledge base...")
    await msg.send()

    # Make API request with chat history
    search_url = f"{API_BASE_URL}/chat"
    print(cl.user_session.get("session_id"))
    params = {
        "query": user_message,
        "sessionId": cl.user_session.get("session_id")
        # "top_k": 5,
        # "chat_history": chat_history,  # Send properly formatted history
    }
    try:
        async with aiohttp.ClientSession() as session:
            # async with session.get(search_url, params=params) as response:
            async with session.post(search_url, params=params) as response:
                if response.status == 200:
                    qa_results = await response.json()
                    answer_content = f"{qa_results['answer']}\n\n"

                    # Display sources if available
                    if "sources" in qa_results and qa_results["sources"]:
                        answer_content += "### Sources:\n\n"
                        for i, source in enumerate(qa_results["sources"], 1):
                            answer_content += f"**Source {i}**\n"
                            if "document_id" in source:
                                answer_content += f"Document: {source['document_id']}\n"
                            if "page_number" in source:
                                answer_content += f"Page: {source['page_number']}\n"
                            if "content" in source:
                                answer_content += f"Content: {source['content']}\n\n"

                    # Remove loading message and send the actual content
                    await msg.remove()
                    await cl.Message(content=answer_content).send()

                else:
                    error_text = await response.text()
                    await msg.remove()
                    await cl.Message(content=f"⚠️ I encountered an error while searching: {error_text}").send()

    except Exception as e:
        await msg.remove()
        await cl.Message(content=f"⚠️ I encountered an error while trying to answer your question: {str(e)}").send()
