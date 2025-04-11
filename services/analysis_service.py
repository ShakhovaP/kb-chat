import logging
import json
import os
from typing import List, Dict, Tuple, Optional, Any, Union
import io
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_mongodb import MongoDBChatMessageHistory

from services.config_manager import ConfigManager
from services.azure_service import AzureServiceClients

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalysisService:
    """Service for analyzing customer feedback and generating insights."""

    # Predefined categories for customer feedback insights
    PREDEFINED_TOPICS = [
        "turnaround time", 
        "technical expertise", 
        "quality of communication", 
        "responsiveness",
        "reliability",
    ]
    
    def __init__(self):
        """Initialize the analysis service with required dependencies."""
        self.config_manager = ConfigManager()
        self.azure_services = AzureServiceClients(self.config_manager)
        self.openai_deployment = self.config_manager.get_required_env("AZURE_OPENAI_GPT_DEPLOYMENT")
        self.mongodb_uri = self.config_manager.get_required_env("MONGODB_URI")
        
        # Initialize NLP models
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.bertopic_model = BERTopic(embedding_model=self.sentence_model, calculate_probabilities=True)

    def read_excel(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Read customer feedback data from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            DataFrame with Score and Comments columns, or None if there's an error
        """
        try:
            df = pd.read_excel(file_path)
            scores_comments_df = df[['Score', 'Comments']]
            scores_comments_df = scores_comments_df.dropna()
            return scores_comments_df
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            logger.error("Make sure the provided Excel file contains 'Score' and 'Comments' columns")
            return None

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

    def extract_insights(self, comment: str) -> List[Dict]:
        """
        Extract insights from a customer comment using Azure OpenAI.
        
        Args:
            comment: Customer comment text
            
        Returns:
            List of insights with satisfaction indicators
        """
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
        
        try:
            response = self.azure_services.azure_openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            insights_data = json.loads(content)
            
            # Handle different possible response structures
            if isinstance(insights_data, list):
                return insights_data
            elif "insights" in insights_data and "isSatisfied" in insights_data:
                return insights_data
            elif isinstance(insights_data, dict) and any(isinstance(insights_data.get(k), list) for k in insights_data):
                # Find the first list in the response
                for k, v in insights_data.items():
                    if isinstance(v, list):
                        return v
            else:
                # Create a simple structure if format is unexpected
                return [{"insight": content.strip(), "isSatisfied": False}]
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing OpenAI response: {e}")
            logger.error(f"Raw response: {content}")
            return [{"insight": "Error parsing response", "isSatisfied": False}]
        except Exception as e:
            logger.error(f"Unexpected error in extract_insights: {e}")
            return [{"insight": "Error processing comment", "isSatisfied": False}]

    def determine_topic(self, insight_text: str, similarity_threshold: float = 0.5) -> Tuple[str, bool]:
        """
        Determine the topic category for an insight.
        
        Args:
            insight_text: The insight text to categorize
            similarity_threshold: Threshold for matching with predefined topics
            
        Returns:
            Tuple of (topic_name, is_new_topic)
        """
        # Encode the insight and predefined topics
        insight_embedding = self.sentence_model.encode([insight_text])[0]
        topics_embeddings = self.sentence_model.encode(self.PREDEFINED_TOPICS)
        
        # Calculate cosine similarity
        similarities = cosine_similarity([insight_embedding], topics_embeddings)[0]
        
        # Find the most similar topic
        max_similarity_index = np.argmax(similarities)
        max_similarity = similarities[max_similarity_index]
        
        if max_similarity >= similarity_threshold:
            return self.PREDEFINED_TOPICS[max_similarity_index], False
        else:
            # Generate new topic with OpenAI
            return self._generate_new_topic(insight_text)

    def _generate_new_topic(self, insight_text: str) -> Tuple[str, bool]:
        """
        Generate a new topic name for an insight using Azure OpenAI.
        
        Args:
            insight_text: The insight text to categorize
            
        Returns:
            Tuple of (topic_name, is_new_topic=True)
        """
        prompt = f"""
        Create a short, concise topic title (2-4 words) for the following customer feedback insight:
        
        "{insight_text}"
        
        The topic should be a general category that this insight falls under.
        """
        
        try:
            response = self.azure_services.azure_openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=15
            )
            
            if (hasattr(response, 'choices') and 
                len(response.choices) > 0 and
                hasattr(response.choices[0], 'message') and 
                hasattr(response.choices[0].message, 'content')):
                
                new_topic = response.choices[0].message.content
                if new_topic:
                    return new_topic.strip().strip('"'), True
            
            # Fallback if response doesn't have expected structure
            logger.warning(f"Unexpected response structure: {response}")
            return f"Topic for: {insight_text[:20]}...", True
            
        except Exception as e:
            logger.error(f"Error generating topic with OpenAI: {e}")
            return f"Topic for: {insight_text[:20]}...", True

    def process_customer_comments(self, comments: List[str], message_history) -> List[Dict]:
        """
        Process a list of customer comments to extract insights and determine topics.
        
        Args:
            comments: List of customer comment strings
            
        Returns:
            List of processed comments with associated insights
        """
        results = []
        
        # Collect all insights for BERTopic training
        all_insights_text = []
        comments_insights_map = []

        def process_predefined(comment):
            splited_comment = comment.split(", ")
            print("splited_comment", splited_comment)
            if all(item.lower() in self.PREDEFINED_TOPICS for item in splited_comment):
                print("comment", comment)
                for insight in splited_comment:
                    all_insights_text.append(insight + "  needs improvement")
                    comments_insights_map.append({
                        "comment": comment,
                        "insight": insight + "  needs improvement",
                        "isSatisfied": False
                    })
                return True
            return False
        
        for i, comment in enumerate(comments):
            isPredefined = process_predefined(comment)
            print("\n\n", comment, isPredefined)
            if not isPredefined:
                insights_data = self.extract_insights(comment)
                for insight in insights_data:
                    all_insights_text.append(insight["insight"])
                    comments_insights_map.append({
                        "comment": comment,
                        "insight": insight["insight"],
                        "isSatisfied": insight["isSatisfied"]
                    })
        
        # Train BERTopic model on all insights if we have enough data
        topics = None
        if len(all_insights_text) >= 2:  # BERTopic needs at least 2 documents
            topics, _ = self.bertopic_model.fit_transform(all_insights_text)
        
        print("comments_insights_map", comments_insights_map)
        # Process each comment with its insights
        insight_index = 0
        current_comment = None
        comment_results = []
        
        for item in comments_insights_map:
            comment = item["comment"]
            insight_text = item["insight"]
            is_satisfied = item["isSatisfied"]
            
            # Start a new comment result if needed
            if comment != current_comment:
                if current_comment is not None:
                    results.append({
                        "comment": current_comment,
                        "summary": comment_results
                    })
                current_comment = comment
                comment_results = []
            
            # Determine the topic
            if topics and len(all_insights_text) >= 2:
                # Get BERTopic's assigned topic
                bertopic_idx = topics[insight_index]
                
                # Get the topic name
                if bertopic_idx != -1:  # -1 is the outlier topic in BERTopic
                    topic_words = self.bertopic_model.get_topic(bertopic_idx)
                    bertopic_topic = " ".join([word for word, _ in topic_words[:2]])
                else:
                    bertopic_topic = "Miscellaneous"
                    
                # Match with predefined topics or use BERTopic result
                final_topic, is_new = self.determine_topic(insight_text)
            else:
                # For very small datasets, just match with predefined topics
                final_topic, is_new = self.determine_topic(insight_text)
            
            # Add to current comment results
            comment_results.append({
                "topic": final_topic,
                "insight": insight_text,
                "isSatisfied": is_satisfied
            })
            
            insight_index += 1
        
        # Add the last comment
        if current_comment is not None:
            results.append({
                "comment": current_comment,
                "summary": comment_results
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

    def generate_positive_summary(self, insights_df: pd.DataFrame) -> Tuple[str, plt.Figure]:
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
            sum_title="The positive comments say"
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
            prompt=improvement_prompt
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
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str

    def analyze_document(self, file_path: str, session_id: str) -> Dict[str, Any]:
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

        # Calculate NPS distribution
        nps_obj, nps_plot = self.calculate_nps_distribution(df)
        
        # Load test data from file
        try:
            comments_list = list(df['Comments'])
            data = self.process_customer_comments(comments_list, message_history)
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
        
        # Split insights by NPS category
        promoter_insights = insights_df[insights_df["score"] >= 9]
        detractor_insights = insights_df[insights_df["score"] <= 6]
        passive_insights = insights_df[(insights_df["score"] > 6) & (insights_df["score"] < 9)]

        # Generate various summaries and visualizations
        # positive_sum, positive_plot = self.generate_positive_summary(insights_df)
        positive_sum, positive_plot = self.generate_positive_summary(promoter_insights)
        
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