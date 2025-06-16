import chainlit as cl
from bson import ObjectId
from handlers.upload import handle_excel_upload, create_excel_upload_button
from handlers.chat import handle_chat_message

@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    session_id = str(ObjectId())
    cl.user_session.set("session_id", session_id)
    await cl.Message(
        content="""
        Welcome to the Knowledge Base Assistant!      
        You can:
        - Ask questions to the Knowledge Base
        - Upload an Excel file with NPS data for analysis
        - Ask questions about the NPS analysis results
        """
    ).send()
    # await handle_excel_upload()
    await create_excel_upload_button()


@cl.action_callback("upload_excel")
async def on_excel_upload(action):
    """Trigger the Excel upload and NPS processing flow."""
    await handle_excel_upload()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming chat messages."""
    await handle_chat_message(message)
