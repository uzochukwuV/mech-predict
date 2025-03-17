# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 uzo
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""This module contains a Trump news analyser and reporter."""

import functools
import json
import os
import requests
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Optional, Tuple, Callable, Union

import openai
from pydantic import BaseModel, BeforeValidator, Field

# Type definitions
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

# Constants
DEFAULT_OPENAI_MODEL = "Meta-Llama-3-1-8B-Instruct-FP8"
DEFAULT_NEWS_TIMEFRAME_DAYS = 7
DEFAULT_NEWS_SOURCES = ["cnn", "foxnews", "nytimes", "wsj", "politico", "thehill"]

# Key rotation mechanism
def with_key_rotation(func: Callable):
    """Decorator for API key rotation on rate limits."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        # this is expected to be a KeyChain object
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except openai.RateLimitError as e:
                # try with a new key again
                if retries_left["openai"] <= 0 and retries_left["news_api"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                retries_left["news_api"] -= 1
                api_keys.rotate("openai")
                api_keys.rotate("news_api")
                return execute()
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper

# Pydantic models for data validation
class NewsArticle(BaseModel):
    """Model for a news article."""
    source_name: str = Field(..., description="Name of the news source")
    title: str = Field(..., description="Article title")
    description: Optional[str] = Field(None, description="Article description")
    url: str = Field(..., description="Article URL")
    published_at: Optional[str] = Field(None, description="Publication date")
    
    @classmethod
    def from_api(cls, article: Dict[str, Any]) -> "NewsArticle":
        """Create a NewsArticle from API response."""
        return cls(
            source_name=article.get("source", {}).get("name", "Unknown"),
            title=article.get("title", "No title"),
            description=article.get("description", "No description"),
            url=article.get("url", "No URL"),
            published_at=article.get("publishedAt", "Unknown date")
        )

class NewsAnalysis(BaseModel):
    """Model for news analysis results."""
    sentiment: str = Field(..., description="Overall sentiment towards Trump")
    themes: List[str] = Field(..., description="Key themes in the articles")
    events: List[str] = Field(..., description="Significant events mentioned")
    perspectives: List[str] = Field(..., description="Different perspectives represented")
    bias: str = Field(..., description="Any apparent bias in the reporting")

class NewsReport(BaseModel):
    """Model for the final news report."""
    query: str = Field(..., description="The search query used")
    date: str = Field(..., description="Report generation date")
    summary: str = Field(..., description="Executive summary of the news")
    analysis: NewsAnalysis = Field(..., description="Detailed analysis of the news")
    key_articles: List[NewsArticle] = Field(..., description="Key representative articles")
    article_count: int = Field(..., description="Total number of articles analyzed")
    timeframe_days: int = Field(..., description="Time period for news collection")

def fetch_news_articles(
    query: str,
    api_key: str,
    days: int = DEFAULT_NEWS_TIMEFRAME_DAYS
) -> List[NewsArticle]:
    """Fetch news articles related to the query from the News API."""
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    api_url = "https://newsapi.org/v2/everything"
    
    try:
        response = requests.get(
            api_url,
            params={
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'apiKey': api_key,
                'language': 'en',
                'pageSize': 100
            }
        )
        response.raise_for_status()
        data = response.json()
        print(data)
        if data.get('status') == 'ok':
            return [NewsArticle.from_api(article) for article in data.get('articles', [])]
        else:
            print(f"Error fetching from News API: {data.get('message')}")
            return []
            
    except Exception as e:
        print(f"Exception when fetching news: {str(e)}")
        return []

def analyze_news_sentiment(
    articles: List[NewsArticle],
    openai_api_key: str,
    model: str = DEFAULT_OPENAI_MODEL
) -> NewsAnalysis:
    """Analyze sentiment and key themes in news articles using OpenAI."""
    if not articles:
        raise ValueError("No articles to analyze")
    
    # Prepare a sample of articles for analysis
    sample_size = min(20, len(articles))
    sample = articles[:sample_size]
    
    article_texts = [
        f"Source: {article.source_name}\n"
        f"Title: {article.title}\n"
        f"Description: {article.description}\n"
        f"URL: {article.url}\n"
        for article in sample
    ]
    
    combined_text = "\n".join(article_texts)
    
    prompt = f"""
    Analyze the following news articles about Donald Trump:
    
    {combined_text}
    
    Please provide:
    1. Overall sentiment towards Trump (positive, negative, neutral, or mixed)
    2. Key themes and topics discussed
    3. Any significant events mentioned
    4. Different perspectives represented in the articles
    5. Any apparent bias in the reporting
    
    Format your response as a JSON object with these exact keys: 
    - "sentiment" (a string)
    - "themes" (an array of strings)
    - "events" (an array of strings)
    - "perspectives" (an array of strings)
    - "bias" (a string)
    
    Make sure all fields are present and have the correct data types.
    """
    
    client = openai.OpenAI(api_key=openai_api_key, base_url=(
                "https://chatapi.akash.network"
                "/api/v1"
            ))
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure all required fields are present with correct types
        sanitized_result = {
            "sentiment": str(result.get("sentiment", "Mixed")),
            "themes": [str(theme) for theme in result.get("themes", []) if theme] if isinstance(result.get("themes"), list) else ["No themes identified"],
            "events": [str(event) for event in result.get("events", []) if event] if isinstance(result.get("events"), list) else ["No events identified"],
            "perspectives": [str(perspective) for perspective in result.get("perspectives", []) if perspective] if isinstance(result.get("perspectives"), list) else ["No perspectives identified"],
            "bias": str(result.get("bias", "Neutral")) if isinstance(result.get("bias"), str) else "Unable to determine bias"
        }
        
        return NewsAnalysis(**sanitized_result)
    
    except Exception as e:
        print(f"Error analyzing news: {str(e)}")
        # Fallback to default values in case of error
        default_analysis = {
            "sentiment": "Unknown",
            "themes": ["Unable to analyze themes"],
            "events": ["Unable to analyze events"],
            "perspectives": ["Unable to analyze perspectives"],
            "bias": "Unable to analyze bias"
        }
        return NewsAnalysis(**default_analysis)

def generate_news_summary(
    articles: List[NewsArticle],
    analysis: NewsAnalysis,
    openai_api_key: str,
    model: str = DEFAULT_OPENAI_MODEL
) -> str:
    """Generate a comprehensive summary of Trump-related news."""
    if not articles:
        raise ValueError("No articles to summarize")
    
    # Create a prompt that includes both the articles and the analysis
    prompt = f"""
    Based on recent news articles about Donald Trump and the following analysis:
    
    {analysis.model_dump_json(indent=2)}
    
    Generate a comprehensive, balanced summary of the current Trump-related news landscape. The summary should:
    1. Be approximately 500 words
    2. Cover the main events and developments
    3. Present different perspectives fairly
    4. Note any significant patterns in media coverage
    5. Avoid inserting personal opinions or biases
    """
    
    client = openai.OpenAI(api_key=openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        raise

def create_final_report(
    query: str,
    articles: List[NewsArticle],
    analysis: NewsAnalysis,
    summary: str,
    timeframe_days: int = DEFAULT_NEWS_TIMEFRAME_DAYS
) -> NewsReport:
    """Create a final structured report with summary, analysis, and key articles."""
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Select a few representative articles
    key_articles = articles[:5]
    
    return NewsReport(
        query=query,
        date=today,
        summary=summary,
        analysis=analysis,
        key_articles=key_articles,
        article_count=len(articles),
        timeframe_days=timeframe_days
    )

def report_to_markdown(report: NewsReport) -> str:
    """Convert a NewsReport to a markdown string."""
    article_section = "\n\n".join([
        f"Title: {article.title}\n"
        f"Source: {article.source_name}\n"
        f"Published: {article.published_at}\n"
        f"URL: {article.url}\n"
        f"Summary: {article.description}"
        for article in report.key_articles
    ])
    
    themes_list = "\n".join([f"- {theme}" for theme in report.analysis.themes])
    events_list = "\n".join([f"- {event}" for event in report.analysis.events])
    perspectives_list = "\n".join([f"- {perspective}" for perspective in report.analysis.perspectives])
    
    return f"""
# Trump News Report: {report.date}

## Executive Summary

{report.summary}

## Analysis

### Overall Sentiment
{report.analysis.sentiment}

### Key Themes
{themes_list}

### Significant Events
{events_list}

### Media Perspectives
{perspectives_list}

### Media Bias Assessment
{report.analysis.bias}

## Key Articles

{article_section}

## Methodology
This report was generated using AI analysis of {report.article_count} news articles from various sources published in the last {report.timeframe_days} days.

Report generated on {report.date}
"""

def build_run_result(
    report: Union[NewsReport, str],
    error: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Build the standard result tuple for the run function."""
    if error:
        return (
            json.dumps({"error": error}),
            "",
            None,
            None,
        )
    
    if isinstance(report, NewsReport):
        return (
            report.model_dump_json(),
            report_to_markdown(report),
            None,
            None,
        )
    
    return (
        json.dumps({"report": report}),
        "",
        None,
        None,
    )

@with_key_rotation
def run(
    prompt: str,
    api_keys: Any,
    timeframe_days: int = DEFAULT_NEWS_TIMEFRAME_DAYS,
    model: str = DEFAULT_OPENAI_MODEL,
    **kwargs: Any,  # Just to ignore any other arguments passed to the resolver by the benchmark script.
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """
    Run the Trump News Analyzer.
    
    Args:
        prompt: The search query, typically "Donald Trump" or some variant.
        api_keys: An object containing API keys.
        timeframe_days: Number of days to look back for news.
        model: The OpenAI model to use.
        
    Returns:
        A tuple containing (result_json, markdown_report, None, None)
    """
    query = prompt or "Donald Trump"
    openai_api_key = api_keys["openai"]
    news_api_key = api_keys["news_api"]
    
    try:
        # Step 1: Fetch news articles
        articles = fetch_news_articles(query, news_api_key, timeframe_days)
        if not articles:
            return build_run_result(
                report="No articles found", 
                error="No news articles were retrieved"
            )
            
        # Step 2: Analyze sentiment and themes
        analysis = analyze_news_sentiment(articles, openai_api_key, model)
        
        # Step 3: Generate summary
        summary = generate_news_summary(articles, analysis, openai_api_key, model)
        
        # Step 4: Create final report
        report = create_final_report(query, articles, analysis, summary, timeframe_days)
        
        # Step 5: Convert to markdown and return
        return build_run_result(report)
    
    except Exception as e:
        return build_run_result(
            report=f"Error: {str(e)}",
            error=str(e)
        )



# # For command-line usage
# if __name__ == "__main__":
#     import sys
#     from argparse import ArgumentParser
    
#     parser = ArgumentParser(description="Trump News Analyzer")
#     parser.add_argument("--query", type=str, default="Donald Trump", help="Search query")
#     parser.add_argument("--days", type=int, default=DEFAULT_NEWS_TIMEFRAME_DAYS, help="Number of days to look back")
#     parser.add_argument("--model", type=str, default=DEFAULT_OPENAI_MODEL, help="OpenAI model to use")
#     parser.add_argument("--output", type=str, default="trump_news_report.md", help="Output file for the report")
#     parser.add_argument("--openai-key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key")
#     parser.add_argument("--news-key", type=str, default=os.environ.get("NEWS_API_KEY"), help="News API key")
#     args = parser.parse_args()
    
#     # Simple KeyChain class for API key management
#     class KeyChain:
#         def __init__(self, openai_key: str, news_key: str):
#             self.keys = {
#                 "openai": openai_key,
#                 "news_api": news_key,
#             }
            
#         def __getitem__(self, key):
#             return self.keys.get(key, "")
            
#         def max_retries(self):
#             return {"openai": 3, "news_api": 3}
            
#         def rotate(self, key):
#             print(f"Rotating {key} API key")
    
#     if not args.openai_key or not args.news_key:
#         print("Error: Both OpenAI API key and News API key are required.")
#         print("Set them using --openai-key and --news-key arguments or via environment variables.")
#         sys.exit(1)
    
#     api_keys = KeyChain(args.openai_key, args.news_key)
    
#     print(f"Running Trump News Analyzer")
#     data = run(api_keys=api_keys, prompt="what is the latest trump news")
#     print(data)