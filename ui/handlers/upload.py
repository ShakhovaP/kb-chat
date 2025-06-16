import chainlit as cl
import aiohttp
import os
from .utils import generate_summary_messages

# API_BASE_URL = "http://fastapi-backend:8000"
api_base_url = os.getenv("API_BASE_URL")
# API_BASE_URL = "http://localhost:8000"

async def create_excel_upload_button():
    actions = [
        cl.Action(
            name="upload_excel",
            label="Upload Excel for NPS Analysis",
            description="Upload an Excel file with 'Score' and 'Comments' columns",
            payload={},
            collapsed=False
        )
    ]
    await cl.Message(content="Please choose an action:", actions=actions).send()

async def handle_excel_upload():
    files = await cl.AskFileMessage(
        content="Please upload an Excel file with NPS data",
        accept=["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"],
        max_size_mb=5,
        timeout=180,
    ).send()

    if not files:
        await cl.Message(content="No file was uploaded, try again.").send()
        return

    session_id = cl.user_session.get("session_id")
    loading_msg = cl.Message(content="Processing NPS data from Excel file...")
    await loading_msg.send()

    file = files[0]
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            with open(file.path, 'rb') as f:
                form = aiohttp.FormData()
                form.add_field('file', f.read(), filename=file.name, content_type=file.type)

                async with session.post(f"{api_base_url}/excel_upload", params={"session_id": session_id}, data=form) as response:
                    if response.status == 200:
                        result = await response.json()
                        await loading_msg.remove()
                        await generate_summary_messages(result, session_id)
                        cl.user_session.set("processing", False)
                    else:
                        error_text = await response.text()
                        await loading_msg.remove()
                        await cl.Message(content=f"⚠️ Error processing Excel file: {error_text}").send()
        await create_excel_upload_button()
    except Exception as e:
        await loading_msg.remove()
        await cl.Message(content=f"⚠️ Error uploading Excel file: {str(e)}").send()
        await create_excel_upload_button()
