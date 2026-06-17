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
