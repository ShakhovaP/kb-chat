import aiohttp
import chainlit as cl
from .utils import extract_media_urls, create_message_elements
import os

async def handle_chat_message(message: cl.Message):
    user_message = message.content.strip()
    msg = cl.Message(content="Searching the knowledge base...")
    await msg.send()

    params = {
        "query": user_message,
        "sessionId": cl.user_session.get("session_id")
    }

    try:
        async with aiohttp.ClientSession() as session:
            api_url = os.getenv("API_BASE_URL")
            async with session.post(f"{api_url}/chat", params=params) as response:
                if response.status == 200:
                    qa_results = await response.json()
                    answer_content = f"{qa_results['answer']}\n\n"

                    video_url, pptx_url, pptx_name = extract_media_urls(qa_results['sources'])
                    elements = create_message_elements(video_url, pptx_url, pptx_name)

                    await msg.remove()
                    await cl.Message(
                        content=answer_content,
                        elements=elements,
                    ).send()
                else:
                    error_text = await response.text()
                    await msg.remove()
                    await cl.Message(content=f"⚠️ I encountered an error while searching: {error_text}").send()
    except Exception as e:
        await msg.remove()
        await cl.Message(content=f"⚠️ I encountered an error while trying to answer your question: {str(e)}").send()