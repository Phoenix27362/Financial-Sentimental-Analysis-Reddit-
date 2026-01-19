import praw
import pandas as pd
import numpy as np
from datetime import datetime
import time
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib

# Download NLTK data
print("Downloading required NLTK data...")
for package in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    nltk.download(package, quiet=True)

print("=" * 80)
print("COMPLETE FINANCIAL SENTIMENT ANALYSIS PIPELINE")
print("With Detailed NLP Preprocessing (Tokenization, Lemmatization, Stopwords)")
print("=" * 80)


SUBREDDITS = [
    'stocks', 'investing', 'wallstreetbets', 'StockMarket', 'options',
    'pennystocks', 'Daytrading', 'SecurityAnalysis', 'dividends', 'ValueInvesting',
    'CanadianInvestor', 'UKInvesting', 'Fire', 'Bogleheads', 'Trading',
    'Forex', 'CryptoCurrency', 'CryptoMarkets', 'ETFs'
]

POSTS_PER_SUBREDDIT = 300
COMMENTS_PER_POST = 5
PROGRESS_FILE = 'reddit_progress.csv'
FINAL_CSV = 'reddit_sentiment_data.csv'

BULLISH_KEYWORDS = [
    'bullish', 'bull', 'buy', 'buying', 'bought', 'long', 'calls',
    'moon', 'rocket', '🚀', 'surge', 'rally', 'breakout', 'pump',
    'soar', 'climb', 'rise', 'gain', 'gains', 'profit', 'strong',
    'upgrade', 'uptrend', 'undervalued', 'growth', 'opportunity'
]

BEARISH_KEYWORDS = [
    'bearish', 'bear', 'sell', 'selling', 'short', 'puts',
    'crash', 'dump', 'tank', 'plunge', 'collapse', 'fall',
    'drop', 'decline', 'loss', 'weak', 'downgrade', 'downtrend',
    'overvalued', 'bubble', 'risk', 'recession', 'avoid'
]


# DATA COLLECTION

def collect_reddit_data():

    print("\n" + "=" * 80)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 80)

    # Check for existing progress
    try:
        df_existing = pd.read_csv(PROGRESS_FILE)
        completed_subs = df_existing['subreddit'].unique().tolist()
        all_data = df_existing.to_dict('records')
        print(f"\n✅ Found {len(df_existing)} existing samples")
        print(f"   Already completed: {len(completed_subs)} subreddits")
    except FileNotFoundError:
        completed_subs = []
        all_data = []
        print("\n📝 Starting fresh collection")

    remaining_subs = [s for s in SUBREDDITS if s not in completed_subs]

    if not remaining_subs:
        print("\n🎉 All subreddits already collected!")
        return pd.DataFrame(all_data)

    print(f"\nRemaining subreddits: {len(remaining_subs)}")
    print(f"Expected new samples: ~{len(remaining_subs) * POSTS_PER_SUBREDDIT * (1 + COMMENTS_PER_POST)}")

    # Reddit authentication
    print("\n🔐 Reddit Authentication:")
    print("Get credentials from: https://www.reddit.com/prefs/apps")

    CLIENT_ID = os.getenv('REDDIT_CLIENT_ID') or input("CLIENT_ID: ").strip()
    CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET') or input("CLIENT_SECRET: ").strip()

    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent='FinancialSentiment/3.0'
        )
        reddit.subreddit('python').hot(limit=1)
        print("✅ Connected to Reddit API")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return None

    # Collect data
    print(f"\n📊 Collecting from {len(remaining_subs)} subreddits...")

    for i, sub_name in enumerate(remaining_subs, 1):
        print(f"\n[{i}/{len(remaining_subs)}] r/{sub_name}...", end=' ', flush=True)

        try:
            subreddit = reddit.subreddit(sub_name)
            post_count = 0
            comment_count = 0

            for post in subreddit.top(time_filter='year', limit=POSTS_PER_SUBREDDIT):
                all_data.append({
                    'type': 'post',
                    'subreddit': sub_name,
                    'title': post.title,
                    'text': post.selftext,
                    'full_text': f"{post.title} {post.selftext}",
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created': datetime.fromtimestamp(post.created_utc),
                    'id': post.id
                })
                post_count += 1

                try:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list()[:COMMENTS_PER_POST]:
                        if hasattr(comment, 'body') and len(comment.body) > 20:
                            all_data.append({
                                'type': 'comment',
                                'subreddit': sub_name,
                                'title': '',
                                'text': comment.body,
                                'full_text': comment.body,
                                'score': comment.score,
                                'num_comments': 0,
                                'created': datetime.fromtimestamp(comment.created_utc),
                                'id': comment.id
                            })
                            comment_count += 1
                except:
                    pass

            print(f"✓ {post_count} posts + {comment_count} comments")
            pd.DataFrame(all_data).to_csv(PROGRESS_FILE, index=False)
            print(f"      💾 Total: {len(all_data)} samples saved")
            time.sleep(2)

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    print(f"\n✅ Collection complete! Total: {len(all_data)} samples")
    return pd.DataFrame(all_data)


# DATA CLEANING

def clean_and_label(df):

    print("\n" + "=" * 80)
    print("PHASE 2: CLEANING & LABELING")
    print("=" * 80)

    initial_count = len(df)
    print(f"\nStarting with: {initial_count} samples")

    # Text cleaning
    print("\n🧹 Cleaning text...")
    df['full_text'] = df['full_text'].astype(str)
    df['full_text'] = df['full_text'].str.lower()
    df['full_text'] = df['full_text'].apply(lambda x: re.sub(r'http\S+', '', x))
    df['full_text'] = df['full_text'].apply(lambda x: re.sub(r'\S+@\S+', '', x))
    df['full_text'] = df['full_text'].apply(lambda x: re.sub(r'[^\w\s$%.,!?-]', ' ', x))
    df['full_text'] = df['full_text'].apply(lambda x: ' '.join(x.split()))

    # Remove duplicates and short texts
    df = df.drop_duplicates(subset=['full_text'])
    df = df[df['full_text'].str.len() >= 30]
    df = df[~df['full_text'].str.contains('deleted|removed', case=False, na=False)]

    print(f"   After cleaning: {len(df)} samples ({initial_count - len(df)} removed)")

    # Sentiment labeling
    print("\n🏷️  Labeling sentiment...")

    def label_sentiment(row):
        text = row['full_text']
        score = row.get('score', 0)

        bull_count = sum(1 for word in BULLISH_KEYWORDS if word in text)
        bear_count = sum(1 for word in BEARISH_KEYWORDS if word in text)

        if bull_count > bear_count and bull_count >= 1:
            return 'positive'
        elif bear_count > bull_count and bear_count >= 1:
            return 'negative'
        elif score > 100:
            return 'positive'
        elif score < -5:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment'] = df.apply(label_sentiment, axis=1)

    # Keep only binary sentiments
    df_binary = df[df['sentiment'].isin(['positive', 'negative'])].copy()

    print(f"\n📊 Sentiment distribution:")
    for sent, count in df['sentiment'].value_counts().items():
        print(f"   {sent}: {count:,} ({count / len(df) * 100:.1f}%)")

    print(f"\n✅ Binary dataset: {len(df_binary)} samples")
    print(f"   Positive: {sum(df_binary['sentiment'] == 'positive'):,}")
    print(f"   Negative: {sum(df_binary['sentiment'] == 'negative'):,}")

    return df_binary


# NLP Preprocessing

def detailed_preprocessing(df):

    print("\n" + "=" * 80)
    print("PHASE 3: DETAILED NLP PREPROCESSING")
    print("=" * 80)

    print("\nThis phase includes:")
    print("  1. Tokenization (word_tokenize)")
    print("  2. Stopword removal (keeping financial terms)")
    print("  3. Lemmatization (WordNetLemmatizer)")

    # Tokenization
    print("\n[1/3] Tokenization...")
    df['tokens'] = df['full_text'].apply(word_tokenize)
    token_counts = df['tokens'].apply(len)
    print(f"   ✓ Tokenized all texts")
    print(f"   Mean tokens per text: {token_counts.mean():.1f}")

    # Stopword removal
    print("\n[2/3] Stopword removal...")
    stop_words = set(stopwords.words('english'))

    # Keep financially meaningful words
    financial_keep_words = {
        'up', 'down', 'over', 'under', 'above', 'below',
        'more', 'less', 'high', 'low', 'off', 'on',
        'not', 'no', 'nor', 'against'
    }
    stop_words = stop_words - financial_keep_words

    def remove_stopwords(tokens):
        return [t for t in tokens if t not in stop_words and len(t) > 2]

    df['tokens_filtered'] = df['tokens'].apply(remove_stopwords)

    original_count = df['tokens'].apply(len).sum()
    filtered_count = df['tokens_filtered'].apply(len).sum()
    reduction = (original_count - filtered_count) / original_count * 100

    print(f"   ✓ Removed stopwords")
    print(f"   Token reduction: {reduction:.1f}%")
    print(f"   Kept financial terms: {financial_keep_words}")

    # Lemmatization
    print("\n[3/3] Lemmatization...")
    lemmatizer = WordNetLemmatizer()

    def lemmatize_tokens(tokens):
        return [lemmatizer.lemmatize(token) for token in tokens]

    df['tokens_lemmatized'] = df['tokens_filtered'].apply(lemmatize_tokens)

    # Join back to text
    df['text_preprocessed'] = df['tokens_lemmatized'].apply(lambda x: ' '.join(x))

    print(f"   ✓ Lemmatization complete")

    # Vocabulary statistics
    all_tokens_original = [t for tokens in df['tokens'] for t in tokens]
    all_tokens_final = [t for tokens in df['tokens_lemmatized'] for t in tokens]

    vocab_original = len(set(all_tokens_original))
    vocab_final = len(set(all_tokens_final))

    print(f"\n📊 Preprocessing Statistics:")
    print(f"   Original vocabulary: {vocab_original:,} unique tokens")
    print(f"   Final vocabulary: {vocab_final:,} unique tokens")
    print(f"   Vocabulary reduction: {(1 - vocab_final / vocab_original) * 100:.1f}%")

    return df


# TRAINING AND EVALUATION

def train_and_evaluate(df):


    print("\n" + "=" * 80)
    print("PHASE 4: MODEL TRAINING & EVALUATION")
    print("=" * 80)

    # Feature extraction with TF-IDF on preprocessed text
    print("\n📊 Extracting TF-IDF features from preprocessed text...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(df['text_preprocessed'])
    y = df['sentiment']

    print(f"   Feature matrix: {X.shape}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # Train/test split
    print("\n✂️  Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training: {X_train.shape[0]:,} | Testing: {X_test.shape[0]:,}")

    # Train models
    print("\n🤖 Training models...")

    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'SVM': SVC(kernel='linear', class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    }

    results = {}

    for name, model in models.items():
        print(f"   Training {name}...", end=' ')
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results[name] = {
            'model': model,
            'predictions': pred,
            'accuracy': acc
        }
        print(f"Accuracy: {acc:.3f} ✓")

    # Calculate metrics
    print("\n📈 Calculating metrics...")

    metrics_data = []
    for name, result in results.items():
        y_pred = result['predictions']
        metrics_data.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, pos_label='positive'),
            'Recall': recall_score(y_test, y_pred, pos_label='positive'),
            'F1-Score': f1_score(y_test, y_pred, pos_label='positive')
        })

    df_metrics = pd.DataFrame(metrics_data)
    df_metrics = df_metrics.sort_values('F1-Score', ascending=False)

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)
    print(df_metrics.to_string(index=False))

    # Detailed classification reports
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORTS")
    print("=" * 80)
    for name, result in results.items():
        print(f"\n{name}:")
        print(classification_report(y_test, result['predictions']))

    # Save best model
    best_model_name = df_metrics.iloc[0]['Model']
    best_model = results[best_model_name]['model']

    print(f"\n💾 Saving best model: {best_model_name}")
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return results, df_metrics, y_test, vectorizer


# VISUALIZATION

def create_visualizations(results, df_metrics, y_test, df):
    """Generate all visualization plots"""

    print("\n" + "=" * 80)
    print("PHASE 5: CREATING VISUALIZATIONS")
    print("=" * 80)

    # 1. Confusion Matrices
    print("\n📊 Generating confusion matrices...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['predictions'], labels=['positive', 'negative'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Positive', 'Negative'],
                    yticklabels=['Positive', 'Negative'])
        axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}',
                            fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('1_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: 1_confusion_matrices.png")
    plt.close()

    # 2. Model Comparison
    print("\n📊 Generating model comparison chart...")
    plt.figure(figsize=(12, 6))

    x = np.arange(len(df_metrics))
    width = 0.2

    plt.bar(x - 1.5 * width, df_metrics['Accuracy'], width, label='Accuracy', color='#3498db')
    plt.bar(x - 0.5 * width, df_metrics['Precision'], width, label='Precision', color='#e74c3c')
    plt.bar(x + 0.5 * width, df_metrics['Recall'], width, label='Recall', color='#2ecc71')
    plt.bar(x + 1.5 * width, df_metrics['F1-Score'], width, label='F1-Score', color='#f39c12')

    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, df_metrics['Model'], rotation=45, ha='right')
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('2_model_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: 2_model_comparison.png")
    plt.close()

    # 3. Word Clouds
    print("\n📊 Generating word clouds...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    pos_text = ' '.join(df[df['sentiment'] == 'positive']['text_preprocessed'].head(1000))
    wc_pos = WordCloud(width=800, height=400, background_color='white',
                       colormap='Greens', max_words=100).generate(pos_text)
    ax1.imshow(wc_pos, interpolation='bilinear')
    ax1.set_title('Positive Sentiment Words', fontsize=16, fontweight='bold')
    ax1.axis('off')

    neg_text = ' '.join(df[df['sentiment'] == 'negative']['text_preprocessed'].head(1000))
    wc_neg = WordCloud(width=800, height=400, background_color='white',
                       colormap='Reds', max_words=100).generate(neg_text)
    ax2.imshow(wc_neg, interpolation='bilinear')
    ax2.set_title('Negative Sentiment Words', fontsize=16, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('3_wordclouds.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: 3_wordclouds.png")
    plt.close()

    # 4. Sentiment Distribution
    print("\n📊 Generating sentiment distribution...")
    plt.figure(figsize=(8, 6))
    sent_counts = df['sentiment'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    sent_counts.plot(kind='bar', color=colors)
    plt.title('Sentiment Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)

    for i, (sent, count) in enumerate(sent_counts.items()):
        plt.text(i, count + 100, f'{count:,}\n({count / len(df) * 100:.1f}%)',
                 ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('4_sentiment_distribution.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: 4_sentiment_distribution.png")
    plt.close()

    print("\n✅ All visualizations created!")


# Execution

def main():
    """Main execution pipeline"""

    print("\n" + "=" * 80)
    print("SELECT MODE:")
    print("=" * 80)
    print("1. Full pipeline (collect + preprocess + train + visualize)")
    print("2. Skip collection (use existing data)")
    print("3. Only collect data")

    choice = input("\nEnter choice (1/2/3) [default: 2]: ").strip() or '2'

    # Phase 1: Data Collection
    if choice in ['1', '3']:
        df = collect_reddit_data()
        if df is None:
            print("❌ Data collection failed!")
            return
        df_clean = clean_and_label(df)
        df_clean.to_csv(FINAL_CSV, index=False)
        print(f"\n💾 Saved: {FINAL_CSV}")

        if choice == '3':
            print("\n✅ Data collection complete! Run again with option 2 to train models.")
            return
    else:
        # Load existing data
        try:
            df_clean = pd.read_csv(FINAL_CSV)
            print(f"\n✅ Loaded {len(df_clean)} samples from {FINAL_CSV}")
        except FileNotFoundError:
            print(f"❌ File not found: {FINAL_CSV}")
            print("Run with option 1 or 3 to collect data first.")
            return

    # Phase 3: Detailed Preprocessing
    df_preprocessed = detailed_preprocessing(df_clean)

    # Save preprocessed data with all columns (matches your report)
    df_preprocessed.to_csv('reddit_preprocessed_detailed.csv', index=False)
    print(f"\n💾 Saved: reddit_preprocessed_detailed.csv (with all preprocessing steps)")

    # Phase 4: Model Training
    results, df_metrics, y_test, vectorizer = train_and_evaluate(df_preprocessed)

    # Save metrics
    df_metrics.to_csv('model_metrics.csv', index=False)
    print(f"\n💾 Saved: model_metrics.csv")

    # Phase 5: Visualizations
    create_visualizations(results, df_metrics, y_test, df_preprocessed)

    # Final Summary
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 80)

    print(f"\n📁 Generated Files:")
    print(f"   1. {FINAL_CSV} - Cleaned dataset")
    print(f"   2. reddit_preprocessed_detailed.csv - With all NLP steps")
    print(f"   3. best_model.pkl - Trained model")
    print(f"   4. vectorizer.pkl - TF-IDF vectorizer")
    print(f"   5. model_metrics.csv - Performance metrics")
    print(f"   6. 1_confusion_matrices.png")
    print(f"   7. 2_model_comparison.png")
    print(f"   8. 3_wordclouds.png")
    print(f"   9. 4_sentiment_distribution.png")

    best = df_metrics.iloc[0]
    print(f"\n🏆 Best Model: {best['Model']}")
    print(f"   Accuracy:  {best['Accuracy']:.4f}")
    print(f"   Precision: {best['Precision']:.4f}")
    print(f"   Recall:    {best['Recall']:.4f}")
    print(f"   F1-Score:  {best['F1-Score']:.4f}")

    print("\n📝 This code matches your report methodology:")
    print("   ✓ Manual tokenization with NLTK")
    print("   ✓ Stopword removal (keeping financial terms)")
    print("   ✓ Lemmatization with WordNetLemmatizer")
    print("   ✓ TF-IDF feature extraction")
    print("   ✓ Multiple ML models comparison")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()