#!/bin/bash
# usage: DATA=<dir> BASE=... EXTRA="..." bash webtax/run_train.sh  (tox/train_tox.py with the 48 topic/format outputs, then webtax/eval_webtax.py)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
export PYTHONDONTWRITEBYTECODE=1
L=topic_adult_content,topic_art_design,topic_software_development,topic_crime_law,topic_education_jobs,topic_electronics_hardware,topic_entertainment,topic_social_life,topic_fashion_beauty,topic_finance_business,topic_food_dining,topic_games,topic_health,topic_history_geography,topic_home_hobbies,topic_industrial,topic_literature,topic_politics,topic_religion,topic_science_math_technology,topic_software,topic_sports_fitness,topic_transportation,topic_travel_tourism,format_academic_writing,format_content_listing,format_creative_writing,format_customer_support_page,format_discussion_forum_comment_section,format_faqs,format_incomplete_content,format_knowledge_article,format_legal_notices,format_listicle,format_news_article,format_nonfiction_writing,format_organizational_about_page,format_organizational_announcement,format_personal_about_page,format_personal_blog,format_product_page,format_q_a_forum,format_spam_ads,format_structured_data,format_technical_writing,format_transcript_interview,format_tutorial_how_to_guide,format_user_reviews
python tox/train_tox.py --labels $L --max_len 512 --base ${BASE:-jhu-clsp/mmBERT-small} --data $DATA --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python webtax/eval_webtax.py --models $OUTPUT_DIR/model --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
