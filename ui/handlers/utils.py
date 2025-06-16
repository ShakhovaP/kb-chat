import aiohttp
import chainlit as cl
from urllib.parse import urlparse
import os


def extract_media_urls(sources: list, max_sources: int = 5):
    video_url, pptx_url, pptx_name = None, None, "Power Point"
    for source in sources[:max_sources]:
        if '.mp4' in source:
            video_url = source
        elif '.pptx' in source:
            pptx_url = source
            path = urlparse(source).path
            if path and path != '/':
                filename = path.rstrip('/').split('/')[-1]
                clean_filename = filename.replace('-', ' ').replace('_', ' ')
                pptx_name = clean_filename
    return video_url, pptx_url, pptx_name


def create_message_elements(video_url: str = None, pptx_url: str = None, pptx_name: str = None):
    elements = []
    if video_url:
        elements.append(cl.Video(url=video_url, display="inline"))
    if pptx_url:
        elements.append(cl.File(name=pptx_name, url=pptx_url, display="inline"))
    return elements if elements else None


async def generate_summary_messages(result, session_id):
    nps_categories = result["nps_categories"]
    total_responses = sum([c["number"] for c in nps_categories])
    overall_nps = result["nps_score"]

    table = (
        "| Category | Score Range | Count | Percentage |\n"
        "|----------|-------------|-------|------------|\n"
        f"| **{nps_categories[0]['name']}** | 9-10 | {nps_categories[0]['number']} | {nps_categories[0]['percentage']:.1f}% |\n"
        f"| **{nps_categories[1]['name']}** | 7-8 | {nps_categories[1]['number']} | {nps_categories[1]['percentage']:.1f}% |\n"
        f"| **{nps_categories[2]['name']}** | 0-6 | {nps_categories[2]['number']} | {nps_categories[2]['percentage']:.1f}% |\n"
        f"| **Total** | - | **{total_responses}** | **100.0%** |"
    )
    await cl.Message(content=f"# NPS Analysis Result\n{table}\n**Overall NPS = {overall_nps:.1f}**").send()

    for category, summary, plot_key in zip([
        "Customer loyalty insights", "Detractors Feedback Analysis", "Passives Feedback Analysis"],
        ["positive_summary", "detract_summary", "passiv_summary"],
        ["positive_plot_base64", "detract_plot_base64", "passiv_plot_base64"]):

        await cl.Message(
            content=f"# {category}\n{result[summary]}",
            elements=[cl.Image(
                url=f"data:image/png;base64,{result[plot_key]}", display="inline", size="large")]
        ).send()

    recommendation_query = f"""
    Based on the following NPS analysis, provide specific recommendations to improve customer satisfaction:

    NPS Score: {overall_nps}

    Promoters ({nps_categories[0]['percentage']:.1f}%): 
    {result['positive_summary']}

    Passives ({nps_categories[1]['percentage']:.1f}%): 
    {result['passiv_summary']}

    Detractors ({nps_categories[2]['percentage']:.1f}%): 
    {result['detract_summary']}

    Instructions:
    1. Provide 3–5 actionable recommendations.
    2. Use the language detected in summaries.
    3. Do not switch to English unless summaries are in English.
    """

    async with aiohttp.ClientSession() as session:
        api_url = os.getenv("API_BASE_URL")
        async with session.post(f"{api_url}/chat", params={"query": recommendation_query, "sessionId": session_id}) as chat_response:
            if chat_response.status == 200:
                rec = await chat_response.json()
                await cl.Message(content="# Strategic Recommendations").send()
                await cl.Message(content=rec.get("answer", "Unable to generate recommendations")).send()
            else:
                await cl.Message(content="⚠️ Could not generate recommendations.").send()

    await cl.Message(content="✅ NPS analysis complete! You can now ask questions or upload another file.").send()
