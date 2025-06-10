# Ethiopian Banks: Mobile App Review & Sentiment Analysis


![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PostgreSQL](https://img.shields.io/badge/database-PostgreSQL-blue)
![Status: Completed](https://img.shields.io/badge/status-completed-brightgreen)

## 1. Project Overview

This end-to-end data engineering and analysis project dissects over 1,400 user reviews for the mobile banking applications of three of Ethiopia's largest banks: **Commercial Bank of Ethiopia (CBE)**, **Bank of Abyssinia (BOA)**, and **Dashen Bank**.

The project transforms unstructured text from the Google Play Store into structured, actionable intelligence. It employs a modern data stack, including web scraping, advanced NLP for sentiment and thematic analysis, and a relational database (PostgreSQL) for data persistence. The ultimate goal is to provide data-driven recommendations that can guide strategic decisions for app improvement, enhance user experience, and increase customer retention.

---

## 2. Executive Summary & Key Recommendations

The analysis reveals a clear and consistent picture across all three banks: **users demand stability and simplicity above all else.** While each app has unique nuances, the core challenges and opportunities are shared.

### Key Findings

*   **App Instability is the #1 Frustration:** "Reliability & Bugs" is the most dominant theme in negative reviews. Users frequently complain about apps that `crash`, `freeze`, `fail to load`, or are plagued with `errors`.
*   **Account Access is a Major Hurdle:** The second most significant issue is "Account & Login Issues." Users report being `blocked` or `locked out` of their accounts and experiencing failures with One-Time Password (OTP) verification.
*   **A Simple, Clean UI is a Key Differentiator:** When users leave positive reviews, they consistently praise apps that are `easy`, `simple`, and have a `good interface`.
*   **Competitive Edge:** Bank of Abyssinia shows a slightly higher median sentiment and a tighter distribution of scores, suggesting a more consistent and less frustrating user experience compared to its competitors.

### Strategic Recommendations

1.  **Prioritize App Stability (High Priority):** Dedicate development sprints to fixing the most common bugs related to crashes and transaction failures. A functional app is the foundation of user trust.
2.  **Streamline the Login & OTP Experience:** Audit and overhaul the entire login and authentication flow. Improve OTP reliability and add alternative, secure authentication methods like biometrics.
3.  **Invest in Continuous UI/UX Improvements:** Double down on what works. Use positive feedback on "easy" interfaces as a guide for future development and simplify transaction flows.

---

## 3. Visual Insights & Key Data Points

While the visualizations are available in the `/visuals` directory, their key takeaways are summarized here:

> #### Sentiment Score Distribution
> A comparative box plot of sentiment scores reveals that **Bank of Abyssinia** has a slightly higher median sentiment and a more compact range of scores. In contrast, **CBE** and **Dashen Bank** exhibit a wider distribution with a longer tail towards extremely negative scores, indicating that a larger portion of their users are having a very frustrating experience.

> #### Dominant Review Themes
> A frequency analysis of review themes clearly shows that **"Reliability & Bugs"** is the most discussed topic across all banks. This is closely followed by **"Account & Login Issues"** and **"Transaction Performance,"** confirming that core functionality and accessibility are the primary areas of user concern.

> #### Negative Keyword Analysis
> Word clouds generated from negative reviews are dominated by terms like `error`, `problem`, `update`, `fix`, `slow`, and `login`. This provides direct, qualitative evidence that technical failures and access problems are the most frequent sources of user complaints.

---

## 4. Technical Architecture & Methodology

This project follows a multi-stage data pipeline, simulating a real-world data engineering workflow.

### Stage 1: Data Collection & Preprocessing (`scrape_preprocess.py`)
*   **Source:** Google Play Store.
*   **Tooling:** `google-play-scraper` and `pandas`.
*   **Process:** Scraped 1,500+ recent English-language reviews. The raw data was then cleaned by handling missing values, removing duplicates based on `reviewId`, and normalizing date formats.

### Stage 2: NLP Enrichment (`analyze_reviews.py`)
*   **Sentiment Analysis:** Used the `distilbert-base-uncased-finetuned-sst-2-english` model via the Hugging Face `transformers` library to assign a `POSITIVE` or `NEGATIVE` label and a normalized sentiment score (-1.0 to +1.0).
*   **Thematic Analysis:** Employed a rule-based keyword matching system using `spaCy` for lemmatization. Reviews were tagged with themes like "Account & Login Issues," "Reliability & Bugs," etc.

### Stage 3: Data Persistence (`load_to_postgres.py`)
*   **Database:** PostgreSQL.
*   **Schema:** A normalized relational schema with a `banks` table (bank_id, bank_name) and a `reviews` table linked by a foreign key (`bank_id`). The full schema is available in `schema.sql`.
*   **Loading:** The enriched CSV data was loaded into the PostgreSQL database using the `psycopg2` driver.

### Stage 4: Insight & Visualization (`visualize_insights.py`)
*   **Tooling:** `Matplotlib`, `Seaborn`, and `WordCloud`.
*   **Output:** Generated charts and plots and saved them to the `/visuals` directory to visually communicate findings.

---

## 5. Getting Started

**Prerequisites:** Python 3.9+, Git, and a local installation of PostgreSQL.

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/fintech-review-analytics.git
cd fintech-review-analytics
Use code with caution.
Markdown
2. Set Up the Python Environment
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# Or for Windows: .\venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Download the required spaCy model
python -m spacy download en_core_web_sm
Use code with caution.
Bash
3. Set Up the PostgreSQL Database
Connect to your local PostgreSQL instance and create the database and user.
-- Run these commands in your psql shell
CREATE USER bank_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE bank_reviews OWNER bank_user;
Use code with caution.
SQL
Important: Update the DB_CONFIG dictionary in load_to_postgres.py with your credentials.
4. Run the Full Data Pipeline
Execute the scripts in order.
# 1. Scrape and clean reviews from Google Play
python scrape_preprocess.py

# 2. Analyze sentiment and themes
python analyze_reviews.py

# 3. Load the enriched data into PostgreSQL
python load_to_postgres.py

# 4. (Optional) Generate visualizations
python visualize_insights.py
Use code with caution.
Bash
6. Testing and Continuous Integration
This project is equipped with an automated testing and CI pipeline to ensure code quality and reliability.
Running Tests Locally
Tests are written using the pytest framework. To run them locally:
pytest
Use code with caution.
Bash
This command discovers and runs all tests in the tests/ directory, validating the data preprocessing and analysis logic.
Continuous Integration (CI)
We use GitHub Actions for our CI pipeline (defined in .github/workflows/ci.yml). This workflow is triggered on every push and pull_request to the main branch and performs the following automated checks:
Installs Dependencies: Sets up the environment and installs all packages.
Lints Code: Uses flake8 to check for style violations.
Runs Tests: Executes the pytest suite to validate core logic.
7. Repository Structure
.
├── .github/                # CI/CD workflows
│   └── workflows/
│       └── ci.yml
├── tests/                  # Unit tests for the project
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_analysis.py
├── visuals/                # Saved charts and plots
├── .gitignore
├── README.md               # This file
├── requirements.txt        # Main project dependencies
├── requirements-dev.txt    # Development & testing dependencies
├── scrape_preprocess.py
├── analyze_reviews.py
├── load_to_postgres.py
├── visualize_insights.py
└── schema.sql              # PostgreSQL database schema dump
Use code with caution.
8. Ethical Considerations & Limitations
Negativity Bias: User reviews are not a perfectly representative sample. Customers are often more motivated to leave a review after a negative experience.
Language Limitation: This analysis was performed on English-language reviews only, excluding feedback in Amharic or other local languages.
Platform Bias: This data is from the Google Play Store (Android users) only and does not reflect the experience of iOS users on the Apple App Store.