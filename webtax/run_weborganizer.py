"""Baseline: WebOrganizer/TopicClassifier-NoURL and FormatClassifier-NoURL (English, gte-base, trust_remote_code) on webtax eval docs.
python webtax/run_weborganizer.py IN.parquet OUT.parquet   (adds wo_topic, wo_format as our label slugs)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo_fix import patch  # transformers 5 compatibility of the custom gte code
import pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
MAP = {"Adult": "adult_content", "Art & Design": "art_design", "Food & Dining": "food_dining", "Games": "games", "Health": "health", "History": "history_geography",
       "Home & Hobbies": "home_hobbies", "Industrial": "industrial", "Literature": "literature", "Politics": "politics", "Religion": "religion",
       "Science & Tech.": "science_math_technology", "Software Dev.": "software_development", "Software": "software", "Sports & Fitness": "sports_fitness",
       "Transportation": "transportation", "Travel": "travel_tourism", "Crime & Law": "crime_law", "Education & Jobs": "education_jobs",
       "Hardware": "electronics_hardware", "Entertainment": "entertainment", "Social Life": "social_life", "Fashion & Beauty": "fashion_beauty",
       "Finance & Business": "finance_business",
       "Academic Writing": "academic_writing", "Content Listing": "content_listing", "News Article": "news_article", "Nonfiction Writing": "nonfiction_writing",
       "About (Org.)": "organizational_about_page", "News (Org.)": "organizational_announcement", "About (Pers.)": "personal_about_page",
       "Personal Blog": "personal_blog", "Product Page": "product_page", "Q&A Forum": "q_a_forum", "Spam / Ads": "spam_ads", "Structured Data": "structured_data",
       "Creative Writing": "creative_writing", "Documentation": "technical_writing", "Audio Transcript": "transcript_interview", "Tutorial": "tutorial_how_to_guide",
       "User Review": "user_reviews", "Customer Support": "customer_support_page", "Comment Section": "discussion_forum_comment_section", "FAQ": "faqs",
       "Truncated": "incomplete_content", "Knowledge Article": "knowledge_article", "Legal Notices": "legal_notices", "Listicle": "listicle"}
df = pd.read_parquet(sys.argv[1])
for name, col in [("WebOrganizer/TopicClassifier-NoURL", "wo_topic"), ("WebOrganizer/FormatClassifier-NoURL", "wo_format")]:
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True, use_memory_efficient_attention=False, torch_dtype=torch.bfloat16).eval()
    m = patch(m).cuda()
    pred = []
    with torch.no_grad():
        for s in range(0, len(df), 32):
            e = tok(df.text.iloc[s:s + 32].tolist(), return_tensors="pt", truncation=True, max_length=512, padding=True)
            pred += [MAP[m.config.id2label[i]] for i in m(input_ids=e["input_ids"].cuda(), attention_mask=e["attention_mask"].cuda()).logits.argmax(-1).tolist()]
    df[col] = pred; print(name, pd.Series(pred).value_counts().head(5).to_dict(), flush=True)
df.drop(columns=["text"]).to_parquet(sys.argv[2])
