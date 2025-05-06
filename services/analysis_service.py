import asyncio
import logging
import json
import os
from typing import List, Dict, Tuple, Optional, Any, Union
import io
import base64

import httpx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_mongodb import MongoDBChatMessageHistory

from services.config_manager import ConfigManager
from services.azure_service import AzureServiceClients
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalysisService:
    """Service for analyzing customer feedback and generating insights."""
    
    def __init__(self):
        """Initialize the analysis service with required dependencies."""
        self.PREDEFINED_TOPICS = [
            "turnaround time", 
            "technical expertise", 
            "quality of communication", 
            "responsiveness",
            "reliability",
        ]

        self.config_manager = ConfigManager()
        self.azure_services = AzureServiceClients(self.config_manager)
        self.openai_deployment = self.config_manager.get_required_env("AZURE_OPENAI_GPT_DEPLOYMENT")
        self.openai_endpoint = self.config_manager.get_required_env("AZURE_OPENAI_ENDPOINT")
        self.openai_apikey = self.config_manager.get_required_env("AZURE_OPENAI_API_KEY")
        self.mongodb_uri = self.config_manager.get_required_env("MONGODB_URI")
        
        # Initialize NLP models
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.bertopic_model = BERTopic(embedding_model=self.sentence_model, calculate_probabilities=True)

        self.generated_topics = []
        self.speed_list = []

    def read_excel(self, file_path: str) -> pd.DataFrame | dict:
        """
        Read an Excel file and extract score and comment columns regardless of language.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            DataFrame with standardized 'Score' and 'Comments' columns, or
            Dictionary with error information if processing failed
        """
        try:
            # Common column name patterns for scores and comments in different languages
            score_patterns = ['score', 'nps-score', 'scorer', 'punktzahl', 'bewertung', 'poäng', 'vurdering']
            comment_patterns = ['comment', 'kommentar', 'bemerkung', 'notat', 'feedback']
            
            # Read the Excel file
            df = pd.read_excel(file_path)
            
            if df.empty:
                return {"error": "The Excel file is empty"}
            
            # Find score and comment columns by pattern matching
            score_col = None
            comment_col = None
            
            # Try to find columns using pattern matching
            for col in df.columns:
                col_lower = str(col).lower()
                
                # Check if this column matches any score pattern
                if score_col is None and any(pattern in col_lower for pattern in score_patterns):
                    score_col = col
                
                # Check if this column matches any comment pattern
                if comment_col is None and any(pattern in col_lower for pattern in comment_patterns):
                    comment_col = col
            
            # If we didn't find both columns through pattern matching, use AI-based identification
            if score_col is None or comment_col is None:
                # Try up to 3 times to get a valid response from the AI
                target_columns = ["nps score", "user comment"]
                max_attempts = 3
                
                for attempt in range(max_attempts):
                    try:
                        # Send only column information and a textual representation of the data
                        identified_columns = self._parse_identified_columns(
                            self.identify_columns(df.head(5), target_columns)
                        )
                        
                        if identified_columns and len(identified_columns) == 2:
                            score_col = identified_columns[0]
                            comment_col = identified_columns[1]
                            break
                        
                    except Exception as e:
                        logger.warning(f"Column identification attempt {attempt+1} failed: {e}")
                        
                        # On last attempt, give up but don't raise an exception
                        if attempt == max_attempts - 1:
                            break
            
            # Final validation that we have both required columns
            if score_col is None or comment_col is None:
                missing = []
                if score_col is None:
                    missing.append("score")
                if comment_col is None:
                    missing.append("comment")
                return {"error": f"Could not identify {' and '.join(missing)} columns in the file"}
                
            # Extract and rename the relevant columns
            scores_comments_df = df[[score_col, comment_col]].rename(columns={
                score_col: 'Score',
                comment_col: 'Comments'
            })

            # Remove rows with NaN values
            # scores_comments_df = scores_comments_df.dropna()
            scores_comments_df = scores_comments_df.dropna(subset=['Score', 'Comments'], how='all')
            return scores_comments_df
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            return {"error": str(e)}
        
    def identify_columns(self, df: pd.DataFrame, columns: list) -> str:
        """
        Use Azure OpenAI to identify which columns in the dataframe match the descriptions.
        
        Args:
            df: DataFrame containing the columns to identify (sample rows)
            columns: List of column descriptions to match
            
        Returns:
            String response from the AI model
        """
        system_prompt = f"""
        You are an advanced AI system specialized in identifying specific columns within a DataFrame.
        Given a DataFrame and a target list of column descriptions: {", ".join(columns)},
        your task is to return a list of the original DataFrame column names that best match these descriptions, in the same order as provided.
        Return only the matching column names as a Python list literal, like: ["Column1", "Column2"]
        """
        
        try:
            # Create a safe representation of the DataFrame for JSON serialization
            sample_data = {}
            for col in df.columns:
                # Convert values to strings to avoid JSON serialization issues
                safe_values = []
                for val in df[col].head(5).values:
                    if pd.isna(val):
                        safe_values.append("NaN")
                    elif isinstance(val, (int, float)) and (val == float('inf') or val == float('-inf')):
                        safe_values.append(str(val))
                    else:
                        safe_values.append(str(val))
                sample_data[col] = safe_values
            
            response = self.azure_services.azure_openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": (
                        f"Column Names: {list(df.columns)}\n"
                        f"Sample Data (first {len(sample_data[list(df.columns)[0]])} rows):\n"
                        f"{sample_data}\n"
                        f"Target Columns to Identify: {columns}"
                    )}
                ]
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Column identification error: {e}")
            raise
            
    def _parse_identified_columns(self, ai_response: str) -> list:
        """
        Parse the AI response to extract column names as a proper list.
        
        Args:
            ai_response: String response from the AI model
            
        Returns:
            List of identified column names
        """
        try:
            # Try to extract a Python list from the response
            # Handle both properly formatted list literals and text descriptions
            ai_response = ai_response.strip()
            
            # Try direct eval if the response looks like a proper Python list
            if ai_response.startswith('[') and ai_response.endswith(']'):
                try:
                    return eval(ai_response)
                except:
                    pass
            
            # Fall back to regex extraction if eval fails
            import re
            column_match = re.findall(r'["\'](.*?)["\']', ai_response)
            if column_match:
                return column_match
            
            # If no columns were found with the above methods
            logger.warning(f"Could not parse columns from AI response: {ai_response}")
            return []
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return []


    def calculate_nps_distribution(self, df: pd.DataFrame) -> Tuple[List[Dict], plt.Figure]:
        """
        Calculate NPS distribution and create visualization.
        
        Args:
            df: DataFrame with customer scores
            
        Returns:
            Tuple containing NPS category data and matplotlib figure
        """
        n_total = len(df)
        n_detractors = df[df["Score"] <= 6].shape[0]
        n_promoters = df[df["Score"] >= 9].shape[0]
        n_passives = n_total - n_promoters - n_detractors
        
        categories = [
            {
                "name": "Promoters",
                "number": n_promoters,
                "percentage": (n_promoters / n_total) * 100,
                "color": "#4CAF50"
            }, 
            {
                "name": "Passives",
                "number": n_passives,
                "percentage": (n_passives / n_total) * 100,
                "color": "#9E9E9E"
            }, 
            {
                "name": "Detractors",
                "number": n_detractors,
                "percentage": (n_detractors / n_total) * 100,
                "color": "#F44336"
            }
        ]

        # fig = plt.figure(figsize=(6, 6))
        # fig.patch.set_alpha(0.1)
        # plt.pie(
        #     x=[cat["percentage"] for cat in categories], 
        #     labels=[cat["name"] for cat in categories], 
        #     colors=[cat["color"] for cat in categories], 
        #     autopct="%1.1f%%", 
        #     startangle=140
        # )
        # plt.title("NPS Distribution")
        fig = plt.figure(figsize=(10, 8), dpi=100)
        fig.patch.set_alpha(0.0)

        # Create the pie chart with enhanced styling
        wedges, texts, autotexts = plt.pie(
            x=[cat["percentage"] for cat in categories],
            labels=None, 
            colors=[cat["color"] for cat in categories],
            autopct=lambda pct: f"{pct:.1f}%" if pct > 5 else "",
            startangle=140,
            shadow=True,
            explode=[0.05] * len(categories),  # Slightly explode all slices
            textprops={'fontsize': 14},
            wedgeprops={'linewidth': 1, 'edgecolor': 'grey'}
        )

        # Enhance the autopct text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        # Add a legend with category names and percentages
        plt.legend(
            wedges,
            [f"{cat['name']} ({cat['percentage']}%)" for cat in categories],
            title="NPS Categories",
            loc="center left",
            bbox_to_anchor=(0.9, 0, 0.5, 1)
        )

        # Add a styled title
        plt.title(
            "NPS Distribution", 
            fontsize=22, 
            fontweight='bold', 
            pad=20,
            color='#333333'
        )

        # Add subtle grid in the background (optional)
        plt.grid(False)

        # Ensure the plot is tight and there's no wasted space
        plt.tight_layout()
        
        return categories, fig

    async def extract_insights_async(self, comment: str) -> List[Dict]:
        endpoint = f"{self.openai_endpoint}/openai/deployments/{self.openai_deployment}/chat/completions?api-version=2024-02-15-preview"

        prompt = f"""
        Analyze the following customer comment about service improvements:
        "{comment}"
        Extract unique short insights from this comment. For each insight, determine whether the customer is satisfied or not.
        Format your response as a JSON array of objects, where each object has:
        1. "insight": a concise statement of the unique insight (max 10 words)
        2. "isSatisfied": true if the customer is satisfied with this aspect, false if not
        
        Example format:
        [
            {{"insight": "website navigation is confusing", "isSatisfied": false}},
            {{"insight": "customer service was helpful", "isSatisfied": true}}
        ]
        """

        headers = {
            "api-key": self.openai_apikey,
            "Content-Type": "application/json"
        }

        body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(endpoint, headers=headers, json=body)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                logger.debug(f"Raw OpenAI content: {content}")

                parsed = json.loads(content)

                # Validate structure
                if isinstance(parsed, list):
                    if all(isinstance(item, dict) and "insight" in item and "isSatisfied" in item for item in parsed):
                        return parsed
                    else:
                        logger.error("Malformed list content: expected list of dicts with keys")
                elif isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, list):
                            return v

                logger.error(f"Unexpected OpenAI response format: {parsed}")
                return [{"insight": str(parsed), "isSatisfied": False}]

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return [{"insight": "Error parsing OpenAI response", "isSatisfied": False}]
        except Exception as e:
            logger.error(f"Async OpenAI call failed: {e}")
            return [{"insight": "Error processing comment", "isSatisfied": False}]

    async def determine_topic_async(self, insight_text: str, similarity_threshold: float = 0.5) -> Tuple[str, bool]:
        """
        Asynchronously determine the topic category for an insight.
        """
        topics_list = self.PREDEFINED_TOPICS + self.generated_topics
        insight_embedding = self.sentence_model.encode([insight_text])[0]
        topics_embeddings = self.sentence_model.encode(topics_list)

        similarities = cosine_similarity([insight_embedding], topics_embeddings)[0]
        max_similarity_index = int(np.argmax(similarities))
        max_similarity = similarities[max_similarity_index]

        if max_similarity >= similarity_threshold:
            return topics_list[max_similarity_index], False
        else:
            return await self._generate_new_topic_async(insight_text)

    async def _generate_new_topic_async(self, insight_text: str) -> Tuple[str, bool]:
        """
        Asynchronously generate a new topic name for an insight using Azure OpenAI.
        """
        prompt = f"""
        Create a short, concise topic title (2-4 words) for the following customer feedback insight:
        "{insight_text}"
        The topic should be a general category that this insight falls under.
        """

        endpoint = f"{self.openai_endpoint}/openai/deployments/{self.openai_deployment}/chat/completions?api-version=2024-02-15-preview"

        headers = {
            "api-key": self.openai_apikey,
            "Content-Type": "application/json"
        }

        body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 15
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(endpoint, headers=headers, json=body)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                topic = content.strip().strip('"')
                if topic:
                    self.generated_topics.append(topic)
                    return topic, True
                else:
                    return f"insight_text[:20]...", True
        except Exception as e:
            logger.error(f"Error generating topic with OpenAI: {e}")
            return f"{insight_text[:20]}...", True


    async def process_customer_comments_async(self, comments: List[str], message_history) -> List[Dict]:
        results = []
        all_insights_text = []
        comments_insights_map = []

        insights_extraction_start = time.time()

        predefined_comments = []
        non_predefined_comments = []

        def process_predefined(comment):
            splited_comment = comment.split(", ")
            if all(item.lower() in self.PREDEFINED_TOPICS for item in splited_comment):
                for insight in splited_comment:
                    all_insights_text.append(insight + "  needs improvement")
                    comments_insights_map.append({
                        "comment": comment,
                        "insight": insight + "  needs improvement",
                        "isSatisfied": False
                    })
                return True
            return False

        for comment in comments:
            if not process_predefined(comment):
                non_predefined_comments.append(comment)

        # Async OpenAI insight extraction
        tasks = [self.extract_insights_async(comment) for comment in non_predefined_comments]
        insights_results = await asyncio.gather(*tasks)

        for comment, insights_data in zip(non_predefined_comments, insights_results):
            for insight in insights_data:
                all_insights_text.append(insight["insight"])
                comments_insights_map.append({
                    "comment": comment,
                    "insight": insight["insight"],
                    "isSatisfied": insight["isSatisfied"]
                })

        insights_extraction_end = time.time()
        self.speed_list.append({
            "script": "extract insights from comments using OpenAI",
            "time_to_exec": insights_extraction_end - insights_extraction_start
        })

        # BERTopic
        bertopic_categorisation_start = time.time()
        topics = None
        if len(all_insights_text) >= 2:
            topics, _ = self.bertopic_model.fit_transform(all_insights_text)
        bertopic_categorisation_end = time.time()
        self.speed_list.append({
            "script": "categorise using BERTopic",
            "time_to_exec": bertopic_categorisation_end - bertopic_categorisation_start
        })

        topic_determination_start = time.time()

        # Async topic determination
        async def enrich_insight_with_topic(index, item):
            insight_text = item["insight"]
            if topics and len(all_insights_text) >= 2:
                bertopic_idx = topics[index]
                if bertopic_idx != -1:
                    topic_words = self.bertopic_model.get_topic(bertopic_idx)
                    bertopic_topic = " ".join([word for word, _ in topic_words[:2]])
                else:
                    bertopic_topic = "Miscellaneous"

            final_topic, is_new = await self.determine_topic_async(insight_text)
            return {
                "comment": item["comment"],
                "insight": insight_text,
                "isSatisfied": item["isSatisfied"],
                "topic": final_topic
            }

        enriched_tasks = [
            enrich_insight_with_topic(idx, item)
            for idx, item in enumerate(comments_insights_map)
        ]
        enriched_results = await asyncio.gather(*enriched_tasks)

        # Aggregate results per comment
        results = []
        current_comment = None
        comment_results = []

        for enriched in enriched_results:
            comment = enriched["comment"]
            if comment != current_comment:
                if current_comment is not None:
                    results.append({
                        "comment": current_comment,
                        "summary": comment_results
                    })
                current_comment = comment
                comment_results = []

            comment_results.append({
                "topic": enriched["topic"],
                "insight": enriched["insight"],
                "isSatisfied": enriched["isSatisfied"]
            })

        if current_comment is not None:
            results.append({
                "comment": current_comment,
                "summary": comment_results
            })

        topic_determination_end = time.time()
        self.speed_list.append({
            "script": "determine topics with OpenAI",
            "time_to_exec": topic_determination_end - topic_determination_start
        })

        return results

    def collect_insights(self, data: List[Dict]) -> pd.DataFrame:
        """
        Collect insights from processed comments into a DataFrame.
        
        Args:
            data: List of processed comments with associated insights
            
        Returns:
            DataFrame with insights data
        """
        insights_data = []
        for i, comment in enumerate(data):
            for insight in comment['summary']:
                item = {
                    'insight': insight['insight'],
                    'category': insight['topic'],
                    'is_satisfied': insight['isSatisfied'],
                    'comment_id': i,
                    'original_comment': comment["comment"],
                }
                insights_data.append(item)
        return pd.DataFrame(insights_data)

    def generate_summary_with_plot(
        self, 
        insights_df: pd.DataFrame, 
        is_satisfied: bool, 
        plot_title: str, 
        prompt: str,
        customer_type: str, 
        sum_title: Optional[str] = None
    ) -> Tuple[str, plt.Figure]:
        """
        Generate a summary and visualization for a subset of insights.
        
        Args:
            insights_df: DataFrame containing insights
            is_satisfied: Filter for satisfied (True) or unsatisfied (False) insights
            plot_title: Title for the plot
            prompt: Prompt for the LLM to generate summary
            sum_title: Optional title for the summary text
            
        Returns:
            Tuple of (summary_text, matplotlib_figure)
        """
        if insights_df.empty: 
            return (f"You have no {customer_type}.", None)
        # Group insights by category and count occurrences
        block_df = insights_df[insights_df["is_satisfied"] == is_satisfied].groupby("category")["comment_id"].count()
        block_df = block_df.sort_values(ascending=False)
        n_total = len(insights_df["comment_id"].unique())

        # Create visualization
        fig = plt.figure(figsize=(8, 5))
        fig.patch.set_alpha(0.1)
        
        # Plot top 3 categories if available, otherwise all
        plot_data = block_df.iloc[:3] if len(block_df) > 3 else block_df
        plot_data.plot(kind="bar", color="blue", edgecolor="black")
        
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.title(plot_title, fontweight='bold')
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add count labels on top of bars
        for i, count in enumerate(plot_data.values):
            percentage = (count / n_total) * 100
            plt.text(i, count + 0.2, f"{count} ({percentage:.1f}%)", 
                     ha="center", fontsize=11, fontweight="bold")
        plt.tight_layout()

        # Collect insights for summary generation
        single_insights = []
        for category in block_df.index:
            category_insights = insights_df[insights_df["category"] == category]["insight"].tolist()
            if category_insights:
                single_insights.append(category_insights[0])
        
        # Generate summary using Azure OpenAI
        full_prompt = prompt + f"Input:{single_insights}"
        try:
            response = self.azure_services.azure_openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[{"role": "user", "content": full_prompt}],
                response_format={"type": "text"},
                temperature=0.1
            )
            
            summary = response.choices[0].message.content
            title = f"\n{sum_title}:\n" if sum_title else "\n"
            summary = title + summary
            
            return summary, fig
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}", fig

    def generate_positive_summary(
            self, 
            insights_df: pd.DataFrame, 
            customer_type: str
        ) -> Tuple[str, plt.Figure]:
        """
        Generate summary of positive feedback.
        
        Args:
            insights_df: DataFrame containing insights
            
        Returns:
            Tuple of (summary_text, matplotlib_figure)
        """
        positive_prompt = """
            Given the list of positive customer insights about the service.
            Your task is to create a short and concrise professional summary 
            highlighting the key strengths of the service.
            Return output as a single paragraph without any additional explanations or metadata.
        """
        return self.generate_summary_with_plot(
            insights_df=insights_df,
            is_satisfied=True,
            plot_title="Top 3 reasons why customers would recommend the company",
            prompt=positive_prompt,
            sum_title="The positive comments say",
            customer_type=customer_type
        )
    
    def generate_improvement_summary(
        self, 
        insights_df: pd.DataFrame, 
        customer_type: str
    ) -> Tuple[str, plt.Figure]:
        """
        Generate summary of areas needing improvement.
        
        Args:
            insights_df: DataFrame containing insights
            customer_type: Type of customers (Promoters, Passives, or Detractors)
            
        Returns:
            Tuple of (summary_text, matplotlib_figure)
        """
        improvement_prompt = """
            Given the list of negative customer insights about the service.
            Your task is to create a short and concrise professional summary
            highlighting the areas needing improvement.
            Return output as a single paragraph without any additional explanations or metadata.
        """
        return self.generate_summary_with_plot(
            insights_df=insights_df,
            is_satisfied=False,
            plot_title=f"Top 3 Improvement Areas for {customer_type}",
            prompt=improvement_prompt,
            customer_type=customer_type
        )
    
    def _format_analysis_results_for_history(self, results: Dict[str, Any]) -> str:
        """
        Format analysis results as a text string for MongoDB message history.
        
        Args:
            results: Dictionary of analysis results
            
        Returns:
            Formatted string representation of results
        """
        output = []
        
        # Add session ID
        output.append(f"Session ID: {results['sessionId']}")
        
        # Add NPS information
        output.append("\n## NPS Distribution")
        for category in results['nps_categories']:
            output.append(f"{category['name']}: {category['number']} ({category['percentage']:.1f}%)")
        
        # Add positive summary
        output.append("\n## Positive Feedback Summary")
        output.append(results['positive_summary'])
        
        # Add improvement areas
        # output.append("\n## Improvement Areas for Promoters")
        # output.append(results['promoter_improvement_summary'])
        
        output.append("\n## Improvement Areas for Passives")
        output.append(results['passiv_summary'])
        
        output.append("\n## Improvement Areas for Detractors")
        output.append(results['detract_summary'])
        
        return "\n".join(output)
    
    def fig_to_base64(self, fig):
        """Convert matplotlib figure to base64-encoded string"""
        if not fig: return None
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str

    async def analyze_document(self, file_path: str, session_id: str) -> Dict[str, Any]:
        """
        Analyze customer feedback document and generate comprehensive insights.
        
        Args:
            file_path: Path to the Excel file with customer feedback
            session_id: Session ID for MongoDB message history
            
        Returns:
            Dictionary containing analysis results
        """

        # Initialize message history
        message_history = MongoDBChatMessageHistory(
            connection_string=self.mongodb_uri, 
            session_id=session_id
        )

        # Read and validate input data
        df = self.read_excel(file_path)
        if df is None or len(df) == 0:
            error_msg = "Excel file must contain 'Score' and 'Comments' columns with data"
            logger.error(error_msg)
            message_history.add_user_message("analyze excel")
            message_history.add_ai_message(error_msg)  # Using string, not dict
            return {"error": error_msg}
        rows_count = len(df)
        # comments_count = len(df[df['Comments'].astype(str) != "NaN"])
        comments_count = df['Comments'].notna().sum()
        # Calculate NPS distribution
        nps_distribution_start = time.time()
        nps_obj, nps_plot = self.calculate_nps_distribution(df)
        nps_distribution_end = time.time()
        self.speed_list.append({
            "script": "calculate nps distribution",
            "time_to_exec": nps_distribution_end - nps_distribution_start
        })
        
        # Load test data from file
        try:
            # comments_list = list(df['Comments'])
            comments_list = list(df['Comments'].dropna())
            data = await self.process_customer_comments_async(comments_list, message_history)
        except Exception as e:
            error_msg = f"Error loading test data: {str(e)}"
            logger.error(error_msg)
            message_history.add_user_message("analyze excel")
            message_history.add_ai_message(error_msg)  # Using string, not dict
            return {"error": error_msg}
            
        # Collect insights into DataFrame
        insights_df = self.collect_insights(data)
        
        # Add score information to insights DataFrame
        # Create a mapping from comment to score
        comment_to_score = dict(zip(df['Comments'], df['Score']))
        
        # Add score column to insights_df
        def get_score(row):
            return comment_to_score.get(row['original_comment'], None)
        
        insights_df['score'] = insights_df.apply(get_score, axis=1)
        
        analysis_generation_start = time.time()
        # Split insights by NPS category
        promoter_insights = insights_df[insights_df["score"] >= 9]
        detractor_insights = insights_df[insights_df["score"] <= 6]
        passive_insights = insights_df[(insights_df["score"] > 6) & (insights_df["score"] < 9)]

        # Generate various summaries and visualizations
        # positive_sum, positive_plot = self.generate_positive_summary(insights_df)
        positive_sum, positive_plot = self.generate_positive_summary(promoter_insights, "Promoters")
        print("positive_sum ", positive_sum)
        
        # Generate improvement summaries for each category
        # prom_improvement_summary, prom_improvement_plot = self.generate_improvement_summary(
        #     promoter_insights, "Promoters"
        # )
        detr_improvement_summary, detr_improvement_plot = self.generate_improvement_summary(
            detractor_insights, "Detractors"
        )
        pasv_improvement_summary, pasv_improvement_plot = self.generate_improvement_summary(
            passive_insights, "Passives"
        )
        analysis_generation_end = time.time()
        self.speed_list.append({
            "script": "generate analysis",
            "time_to_exec": analysis_generation_end - analysis_generation_start
        })
        # print("speed_list", self.speed_list)
        # speed_df = pd.DataFrame(self.speed_list)
        # speed_df.to_csv(f"exec_time_for_{rows_count}_rows_{comments_count}_comm.csv")
        self.speed_list = []

        # Compile analysis results
        analysis_results = {
            "sessionId": session_id,
            "nps_categories": nps_obj,
            # "nps_plot": nps_plot,
            "nps_plot_base64": self.fig_to_base64(nps_plot),
            "positive_summary": positive_sum,
            "positive_plot_base64": self.fig_to_base64(positive_plot),
            # "promoter_improvement_summary": prom_improvement_summary, 
            # "promot_plot_base64": self.fig_to_base64(prom_improvement_plot),
            "passiv_summary": pasv_improvement_summary, 
            "passiv_plot_base64": self.fig_to_base64(pasv_improvement_plot),
            "detract_summary": detr_improvement_summary, 
            "detract_plot_base64": self.fig_to_base64(detr_improvement_plot),
        }

        
        # Record in message history - convert complex object to string
        message_history.add_user_message("analyze excel")
        formatted_results = self._format_analysis_results_for_history(analysis_results)
        message_history.add_ai_message(formatted_results)  # Using string, not dict
        
        return analysis_results