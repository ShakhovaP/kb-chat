#ui.py
import os
import chainlit as cl
from chainlit.input_widget import TextInput, Slider
import aiohttp
import json
from bson import ObjectId
import matplotlib.pyplot as plt
import asyncio
from urllib.parse import urlparse, quote


# Configuration
# API_BASE_URL = "http://localhost:8000"
API_BASE_URL = "http://fastapi-backend:8000"
print(dir(cl))

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
    
    # Create upload button for Excel files
    upload_msg = await create_excel_upload_button()

async def create_excel_upload_button():
    """Create the Excel upload button"""
    actions = [
        cl.Action(
            name="upload_excel",
            label="Upload Excel for NPS Analysis",
            description="Upload an Excel file with 'Score' and 'Comments' columns",
            payload={},  # Providing an empty payload
            collapsed=False
        )
    ]
    upload_message = await cl.Message(content="Please choose an action:", actions=actions).send()
    return upload_message

@cl.action_callback("upload_excel")
async def on_excel_upload(action):
    """Handle Excel file upload via action button"""
    files = await cl.AskFileMessage(
        content="Please upload an Excel file with NPS data",
        accept=["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"],
        max_size_mb=5,
        timeout=180,
    ).send()
    
    if not files:
        await cl.Message(content="No file was uploaded, try again.").send()
        return
        

    # Show loading message while processing
    session_id = cl.user_session.get("session_id")

    loading_msg = cl.Message(content="Processing NPS data from Excel file...")
    await loading_msg.send()
    # asyncio.create_task(poll_progress(session_id, loading_msg))
    
    # Upload the Excel file to the backend
    file = files[0]
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            with open(file.path, 'rb') as f:
                file_data = f.read()
                form = aiohttp.FormData()
                form.add_field('file', file_data, filename=file.name, content_type=file.type)
                
                async with session.post(
                    f"{API_BASE_URL}/excel_upload",
                    params={"session_id": session_id},
                    data=form
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                    
                        # Remove loading message
                        await loading_msg.remove()
                        
                        # Display NPS analysis results
                        nps_categories = result["nps_categories"]
                        total_responses = sum([c["number"] for c in nps_categories])
                        overall_nps = result["nps_score"]

                        table_markdown = (
                            "| Category | Score Range | Count | Percentage |\n"
                            "|----------|-------------|-------|------------|\n"
                            f"| **{nps_categories[0]['name']}** | 9-10 | {nps_categories[0]['number']} | {nps_categories[0]['percentage']:.1f}% |\n"
                            f"| **{nps_categories[1]['name']}** | 7-8 | {nps_categories[1]['number']} | {nps_categories[1]['percentage']:.1f}% |\n"
                            f"| **{nps_categories[2]['name']}** | 0-6 | {nps_categories[2]['number']} | {nps_categories[2]['percentage']:.1f}% |\n"
                            f"| **Total** | - | **{total_responses}** | **100.0%** |"
                        )
                        content = "# NPS Analysis Result\n" + table_markdown + f"\n**Overall NPS for {total_responses} uploaded respondents = {overall_nps:.1f}**"
                        await cl.Message(content=content).send()


                        # Display the NPS plot from base64 data
                        nps_plot = [
                            cl.Image(
                                url=f"data:image/png;base64,{result['nps_plot_base64']}",
                                display="inline",
                                size="large"
                            ),
                        ]
                        await cl.Message(
                            content="NPS Distribution Chart:",  # Added content
                            elements=nps_plot,
                        ).send()

                        # Display the positive plot from base64 data
                        pos_plot = [
                            cl.Image(
                                url=f"data:image/png;base64,{result['positive_plot_base64']}",
                                display="inline",
                                size="large"
                            ),
                        ]
                        # await cl.Message(
                        #     content="Promoters Feedback Analysis:",
                        #     elements=pos_plot,
                        # ).send()
                        pos_summary = f"""
                        # Customer loyalty insights
                        {result["positive_summary"]}
                        """
                        await cl.Message(
                            content=pos_summary,
                            elements=pos_plot,
                        ).send()
                        # await cl.Message(content=pos_summary).send()

                        # Display the detractor plot from base64 data
                        detract_plot = [
                            cl.Image(
                                url=f"data:image/png;base64,{result['detract_plot_base64']}",
                                display="inline",
                                size="large"
                            ),
                        ]
                        det_summary = f"""
                        # Detractors Feedback Analysis
                        {result["detract_summary"]}
                        """
                        await cl.Message(
                            content=det_summary,
                            elements=detract_plot,
                        ).send()

                        # Display the passive plot from base64 data
                        passiv_plot = [
                            cl.Image(
                                url=f"data:image/png;base64,{result['passiv_plot_base64']}",
                                display="inline",
                                size="large"
                            ),
                        ]
                        pas_summary = f"""
                        # Passives Feedback Analysis
                        {result["passiv_summary"]}
                        """
                        await cl.Message(
                            content=pas_summary,
                            elements=passiv_plot,
                        ).send()

                        # Prepare a detailed query for the chat endpoint
                        recommendation_query = f"""
                        Based on the following NPS analysis, provide specific recommendations  to improve customer satisfaction:
                        
                        NPS Score: {result.get('nps_score', 'N/A')}
                        
                        Promoters ({nps_categories[0]['percentage']:.1f}%): 
                        {result["positive_summary"]}
                        
                        Passives ({nps_categories[1]['percentage']:.1f}%): 
                        {result["passiv_summary"]}
                        
                        Detractors ({nps_categories[2]['percentage']:.1f}%): 
                        {result["detract_summary"]}
                        
                        Instructions:
                        1. Provide 3–5 specific, actionable recommendations based on the summaries above.
                        2. Detect the language of the summaries automatically and use it throughout your response.
                        3. You MUST write your response in the same language used in the summaries above (e.g., Danish, etc.). 
                        Do NOT switch to English, unless all summaries themselves are written in English.
                        4. Focus on improving customer satisfaction by addressing themes mentioned in each category.
                        """
                        # Call the chat endpoint
                        async with session.post(
                            f"{API_BASE_URL}/chat",
                            params={
                                "query": recommendation_query,
                                "sessionId": session_id
                            }
                        ) as chat_response:
                            if chat_response.status == 200:
                                recommendations = await chat_response.json()
                                
                                # Display recommendations
                                await cl.Message(content="# Strategic Recommendations").send()
                                await cl.Message(content=recommendations.get("answer", "Unable to generate recommendations")).send()
                            else:
                                await cl.Message(content="⚠️ Could not generate recommendations from knowledge base.").send()

                        # await cl.Message(content="You can now ask questions about this NPS data!").send()
                        await cl.Message(content="✅ NPS analysis complete! You can now ask questions about this data or upload another file.").send()
                        
                        # Clear the user session of any blocking states
                        cl.user_session.set("processing", False)
                        
                    else:
                        error_text = await response.text()
                        await loading_msg.remove()
                        await cl.Message(content=f"⚠️ Error processing Excel file: {error_text}").send()
                        
            # Recreate the upload button
            await create_excel_upload_button()
        
    except Exception as e:
        await loading_msg.remove()
        # await loading_msg.update(content=f"⚠️ Error uploading Excel file: {str(e)}")
        await cl.Message(content=f"⚠️ Error uploading Excel file: {str(e)}").send()
        await create_excel_upload_button()

def extract_media_urls(sources: list, max_sources: int = 5):
    """
    Args:
        sources: List of source URLs
        max_sources: Maximum number of sources to process
        
    Returns:
        tuple: (video_url, pptx_url, pptx_name)
    """
    video_url = None
    pptx_url = None
    pptx_name = "Power Point"
    
    for source in sources[:max_sources]:
        if '.mp4' in source:
            video_url = source
        elif '.pptx' in source:
            pptx_url = source
            
            # Extract a readable name for the PowerPoint file
            parsed_url = urlparse(source)
            path = parsed_url.path
            
            if path and path != '/':
                filename = path.rstrip('/').split('/')[-1]
                clean_filename = filename.replace('-', ' ').replace('_', ' ')
                if clean_filename:
                    pptx_name = clean_filename
    
    return video_url, pptx_url, pptx_name


def create_message_elements(video_url: str = None, pptx_url: str = None, pptx_name: str = None):
    """
    Args:
        video_url: URL to a video file
        pptx_url: URL to a PowerPoint file
        pptx_name: Display name for the PowerPoint file
        
    Returns:
        list: List of UI elements to display
    """
    elements = []
    
    if video_url:
        elements.append(cl.Video(url=video_url, display="inline"))
    
    if pptx_url:
        elements.append(cl.File(
            name=pptx_name,
            url=pptx_url,
            display="inline"
        ))
    
    return elements if elements else None

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
                    # Extract media URLs from sources
                    video_url, pptx_url, pptx_name = extract_media_urls(qa_results['sources'])
                    
                    # Create UI elements for the response
                    elements = create_message_elements(video_url, pptx_url, pptx_name)
                    
                    # Remove loading message and send the actual content
                    await msg.remove()
                    await cl.Message(
                        content=answer_content,
                        elements=elements,
                    ).send()
                    return

                else:
                    error_text = await response.text()
                    await msg.remove()
                    await cl.Message(content=f"⚠️ I encountered an error while searching: {error_text}").send()

    except Exception as e:
        await msg.remove()
        await cl.Message(content=f"⚠️ I encountered an error while trying to answer your question: {str(e)}").send()