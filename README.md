
## Setup & Execution
Prerequisites: Python 3.8+, git.

1.  **Clone the repository & switch branch:**
    ```bash
     git clone https://github.com/YOUR_USERNAME/ethio_bank_reviews.git
     cd ethio_bank_reviews
     git checkout task-1 
    ```
2.  **Create and Activate Virtual Environment:**
     ```bash
    python -m venv venv
     # Windows
    .\venv\Scripts\activate 
     # macOS/Linux
     source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
     pip install -r requirements.txt
    ```
4.  **Run the Script:**
    The script will scrape the data, clean it, print status/KPIs, and save the CSV.
     ```bash
    python scrape_preprocess.py
     ```
    This creates the `ethiopian_bank_reviews_cleaned.csv` file in the project root. The script's log output will confirm the final count of reviews and the missing data percentage.

## Methodology: Task 1

### Data Collection
*   **Tool:** Python library `google-play-scraper`.
*   **Source:** Google Play Store.
*   **Target Apps & IDs:**
    *   Commercial Bank of Ethiopia: `com.mode.CBE.mobile`
    *   Bank of Abyssinia: `com.boa.mobile`
    *   Dashen Bank: `com.dashenbanksc.mobile`
	*(Note: IDs correspond to the main mobile banking app for each bank, not wallet apps like CBEBirr or Amole)*.
*   **Parameters:** 
	* A target of 500 reviews per app was set to ensure >400 remain after cleaning.
	* Reviews are sorted by `Sort.NEWEST`.
	* Language=`en`, Country=`us` used to maximise review count.
	* Reviews for all star ratings (1-5) were collected.
*   The script iterates through each bank, scrapes the reviews, and concatenates them into a single Pandas DataFrame. Error handling is included if an app ID fails.

 ### Preprocessing
 The raw data undergoes the following cleaning steps:
1.  **Missing Data:** Rows with missing review text (`content`), `score` or timestamp (`at`) are removed. Rows with empty string or whitespace-only reviews are also removed. The percentage of rows with critical missing data in the *initial raw fetch* is calculated for the KPI.
2.  **Duplicate Removal:** Duplicates are removed based on the unique `reviewId` provided by the scraper to ensure each review is counted only once.
 3. **Column Selection & Renaming:** Only relevant columns are kept and renamed:
    * `content` -> `review`
    * `score`  -> `rating`
    * `at`     -> `date`
4.  **Feature Addition**: `bank` name and `source` ('Google Play Store') columns are added.
5.  **Date Normalisation:** The `date` column (timestamp) is converted and formatted to the `YYYY-MM-DD` string format.
 6. **Output:** The final, cleaned DataFrame is saved to `ethiopian_bank_reviews_cleaned.csv` with columns: `review`, `rating`, `date`, `bank`, `source`.

## Output Data
The script generates `ethiopian_bank_reviews_cleaned.csv` with the following structure:

| review | rating | date | bank | source |
| :--- | :--- | :--- | :--- | :--- |
| The app is very slow...| 2 | 2023-10-25 | Commercial Bank of Ethiopia | Google Play Store |
| Easy to use. | 5 | 2023-10-24 | Bank of Abyssinia | Google Play Store |
| Crashes frequently! | 1 | 2023-10-24 | Dashen Bank | Google Play Store |
| ... | ...| ... | ... | ... |

---