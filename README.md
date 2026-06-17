# 🔍 TruthLayer: AI-Powered PDF Fact-Checker

TruthLayer is an AI-powered fact-checking application designed to extract verifiable claims from PDF documents and cross-reference them against real-time web data. The primary objective is to act as a "Truth Layer," exposing outdated statistics, fabricated figures, and incorrect dates in marketing materials or documents.

## 🚀 Features

* **PDF Parsing:** Converts uploaded PDFs into structured Markdown using PyMuPDF.
* **Automated Claim Extraction:** Uses Google Gemini 2.5 Flash to isolate statistical, financial, and date-specific claims.
* **Live Web Grounding:** Integrates the Tavily Search API to fetch the most current web context for every claim.
* **Intelligent Verdict Engine:** Compares the original claims against live contexts to assign strict verdicts (`Verified`, `Inaccurate`, or `False`).
* **Secure Architecture:** Built with Streamlit Secrets to ensure API keys are never exposed on the frontend or GitHub.

## 🛠️ Tech Stack

* **Language:** Python 3.11+
* **Frontend & Backend:** Streamlit
* **AI/LLM:** Google Gemini API (`gemini-2.5-flash`)
* **Search Engine:** Tavily API
* **PDF Processing:** PyMuPDF & PyMuPDF4LLM
* **Data Validation:** Pydantic

## 💻 How to Run Locally

If you want to run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/naavvviiinnnn/TruthLayer.git](https://github.com/naavvviiinnnn/TruthLayer.git)
   cd TruthLayer
2. Install the dependencies:

 ```bash
pip install -r requirements.txt

3. Set up your API Keys securely:
Create a hidden folder named .streamlit in the root of your project, and inside it, create a file named secrets.toml. Add your keys to this file:

Ini, TOML
GEMINI_API_KEY = "your_gemini_api_key_here"
TAVILY_API_KEY = "your_tavily_api_key_here"
(Note: Ensure .streamlit/ is included in your .gitignore file so you do not accidentally push your keys to GitHub).

4. Run the application:

 ```bash
streamlit run app.py
