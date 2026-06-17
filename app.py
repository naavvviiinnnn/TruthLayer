"""TruthLayer — AI-powered PDF fact-checking with live web grounding."""

from __future__ import annotations

from typing import Literal

import pymupdf
import pymupdf4llm
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tavily import TavilyClient

MODEL = "gemini-2.5-flash"
MAX_CLAIMS = 5
TAVILY_TOP_K = 3


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ExtractedClaim(BaseModel):
    original_quote: str = Field(description="Exact quote from the document.")
    search_optimized_summary: str = Field(
        description="Concise search query to verify this claim on the web."
    )
    claim_type: str = Field(
        description="Category: statistical, date, financial, or other verifiable type."
    )


class DocumentClaims(BaseModel):
    claims: list[ExtractedClaim] = Field(
        description=f"Up to {MAX_CLAIMS} highly specific, verifiable claims."
    )


class VerificationResult(BaseModel):
    verdict: Literal["Verified", "Inaccurate", "False"]
    confidence_score: int = Field(ge=0, le=100)
    source_credibility_score: int = Field(
        ge=0, le=100, description="Domain authority of sources in live context."
    )
    explanation: str
    current_real_fact: str = Field(
        description="Current factual data from live context when claim is outdated or wrong."
    )


# ---------------------------------------------------------------------------
# Page config & dark-mode styling
# ---------------------------------------------------------------------------

st.set_page_config(page_title="TruthLayer", layout="wide", page_icon="🔍")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    .tl-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.75rem 2rem;
        margin-bottom: 2rem;
    }

    .tl-header h1 {
        margin: 0;
        font-size: 2.25rem;
        font-weight: 700;
        background: linear-gradient(90deg, #F39C12, #f1c40f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .tl-header p {
        margin: 0.5rem 0 0 0;
        color: #a0aec0;
        font-size: 1rem;
    }

    .tl-badge {
        display: inline-block;
        background: rgba(243, 156, 18, 0.15);
        color: #F39C12;
        border: 1px solid rgba(243, 156, 18, 0.35);
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .tl-metric-card {
        background: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        text-align: center;
    }

    .tl-metric-label {
        color: #a0aec0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
    }

    .tl-metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #F39C12;
    }

    .tl-fact-box {
        background: rgba(243, 156, 18, 0.08);
        border-left: 4px solid #F39C12;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }

    .tl-fact-box strong {
        color: #F39C12;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #252525 100%);
        border-right: 1px solid #333;
    }

    div[data-testid="stSidebar"] .stMarkdown h2 {
        color: #F39C12;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def extract_claims(client: genai.Client, markdown_text: str) -> DocumentClaims:
    prompt = f"""You are a forensic document analyst. Extract up to {MAX_CLAIMS} highly specific,
verifiable claims from the document below. Focus on:
- Statistical claims (percentages, counts, rankings)
- Date-specific assertions (events, deadlines, timelines)
- Financial figures (revenue, funding, valuations, budgets)

For each claim provide:
- original_quote: the exact passage from the document
- search_optimized_summary: a concise web search query to verify it
- claim_type: one of "statistical", "date", "financial", or "other"

Return fewer than {MAX_CLAIMS} if the document lacks verifiable claims. Do not invent claims.

DOCUMENT:
{markdown_text[:80000]}
"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DocumentClaims,
        ),
    )
    return DocumentClaims.model_validate_json(response.text)


def build_live_context(tavily: TavilyClient, query: str) -> str:
    search_response = tavily.search(query=query, max_results=TAVILY_TOP_K)
    results = search_response.get("results", [])[:TAVILY_TOP_K]
    parts: list[str] = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")
        parts.append(f"[Source {i}] {title}\nURL: {url}\n{content}")
    return "\n\n---\n\n".join(parts) if parts else "No live web results found."


VERIFICATION_SYSTEM = """You are TruthLayer's verdict engine. Evaluate the claim SOLELY against the live context.
If the claim contains outdated statistics, wrong dates, or fake funding numbers compared to the live context,
you MUST output 'Inaccurate' and provide the fresh data in current_real_fact.

Rules:
- "Verified": The claim matches current live context data.
- "Inaccurate": The claim is partially correct but uses outdated or slightly wrong figures/dates.
- "False": The claim directly contradicts live context or is fabricated.
- source_credibility_score: Rate 0-100 based on domain authority of sources in live context
  (e.g. .gov, major news outlets, official company sites score higher).
- current_real_fact: Always provide the most current factual data from live context, even if Verified.
"""


def verify_claim(
    client: genai.Client,
    claim: ExtractedClaim,
    live_context: str,
) -> VerificationResult:
    prompt = f"""{VERIFICATION_SYSTEM}

ORIGINAL CLAIM ({claim.claim_type}):
"{claim.original_quote}"

LIVE WEB CONTEXT:
{live_context}
"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VerificationResult,
        ),
    )
    return VerificationResult.model_validate_json(response.text)


def parse_pdf_to_markdown(uploaded_file) -> str:
    pdf_bytes = uploaded_file.read()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return pymupdf4llm.to_markdown(doc)
    finally:
        doc.close()


def render_verdict_banner(verdict: str, explanation: str) -> None:
    if verdict == "Verified":
        st.success(f"**{verdict}** — {explanation}")
    elif verdict == "Inaccurate":
        st.warning(f"**{verdict}** — {explanation}")
    else:
        st.error(f"**{verdict}** — {explanation}")


# ---------------------------------------------------------------------------
# Sidebar — API keys & Conditional Secrets Logic
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔑 API Configuration")
    
    # 1. Attempt to fetch keys from Streamlit Secrets
    gemini_secret = st.secrets.get("GEMINI_API_KEY", None)
    tavily_secret = st.secrets.get("TAVILY_API_KEY", None)

    # 2. If secrets exist, use them silently. Otherwise, show input boxes.
    if gemini_secret and tavily_secret:
        st.success("API Keys configured securely via cloud secrets.")
        gemini_key = gemini_secret
        tavily_key = tavily_secret
    else:
        st.caption("Keys are used only in this session and never stored.")
        gemini_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="AIza...",
            help="Get your key at https://aistudio.google.com/apikey",
        )
        tavily_key = st.text_input(
            "Tavily API Key",
            type="password",
            placeholder="tvly-...",
            help="Get your key at https://tavily.com",
        )

    st.divider()
    st.markdown("### How it works")
    st.markdown(
        """
        1. **Upload** a PDF document
        2. **Extract** verifiable claims via Gemini
        3. **Ground** each claim with live Tavily search
        4. **Verify** against current web data
        """
    )
# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="tl-header">
        <div class="tl-badge">Intelligence Agency · Fact Check</div>
        <h1>TruthLayer</h1>
        <p>Expose outdated statistics, wrong dates, and fabricated figures with live web grounding.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

keys_ready = bool(gemini_key and tavily_key)
if not keys_ready:
    st.info("Enter your Gemini and Tavily API keys in the sidebar to begin.")

uploaded_pdf = st.file_uploader(
    "Upload a PDF to fact-check",
    type=["pdf"],
    disabled=not keys_ready,
    help="PDFs are parsed to Markdown to preserve tables and structure.",
)

# FAILSAFE: Ensure this variable always exists so Python never throws a NameError
run_analysis = False

# Make absolutely sure this block is NOT indented! It should be all the way to the left margin.
run_analysis = st.button(
    "🔍 Run Fact Check",
    type="primary",
    disabled=not (keys_ready and uploaded_pdf),
    use_container_width=True,
)

if run_analysis and uploaded_pdf and gemini_key and tavily_key:
    gemini_client = get_gemini_client(gemini_key)
    tavily_client = TavilyClient(api_key=tavily_key)

    with st.status("Analyzing document…", expanded=True) as status:
        st.write("📄 Parsing PDF to Markdown…")
        markdown_text = parse_pdf_to_markdown(uploaded_pdf)
        st.write(f"Extracted **{len(markdown_text):,}** characters.")

        with st.expander("Preview extracted Markdown", expanded=False):
            st.markdown(markdown_text[:6000] + ("…" if len(markdown_text) > 6000 else ""))

        st.write("🧠 Extracting verifiable claims with Gemini…")
        document_claims = extract_claims(gemini_client, markdown_text)

        if not document_claims.claims:
            status.update(label="No verifiable claims found", state="complete")
            st.warning("No statistical, date, or financial claims were found in this document.")
        else:
            st.write(f"Found **{len(document_claims.claims)}** claim(s). Running live verification…")
            results: list[tuple[ExtractedClaim, VerificationResult, str]] = []

            for idx, claim in enumerate(document_claims.claims, 1):
                st.write(f"🌐 [{idx}/{len(document_claims.claims)}] Searching: *{claim.search_optimized_summary}*")
                live_context = build_live_context(tavily_client, claim.search_optimized_summary)
                verification = verify_claim(gemini_client, claim, live_context)
                results.append((claim, verification, live_context))

            status.update(label="Analysis complete", state="complete")
            st.session_state["tl_results"] = results

if "tl_results" in st.session_state and st.session_state["tl_results"]:
    st.divider()
    st.markdown("## 📊 Verification Results")

    results = st.session_state["tl_results"]
    verified_count = sum(1 for _, v, _ in results if v.verdict == "Verified")
    inaccurate_count = sum(1 for _, v, _ in results if v.verdict == "Inaccurate")
    false_count = sum(1 for _, v, _ in results if v.verdict == "False")

    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.markdown(
            f'<div class="tl-metric-card"><div class="tl-metric-label">Claims</div>'
            f'<div class="tl-metric-value">{len(results)}</div></div>',
            unsafe_allow_html=True,
        )
    with summary_cols[1]:
        st.markdown(
            f'<div class="tl-metric-card"><div class="tl-metric-label">Verified</div>'
            f'<div class="tl-metric-value" style="color:#2ecc71">{verified_count}</div></div>',
            unsafe_allow_html=True,
        )
    with summary_cols[2]:
        st.markdown(
            f'<div class="tl-metric-card"><div class="tl-metric-label">Inaccurate</div>'
            f'<div class="tl-metric-value" style="color:#f39c12">{inaccurate_count}</div></div>',
            unsafe_allow_html=True,
        )
    with summary_cols[3]:
        st.markdown(
            f'<div class="tl-metric-card"><div class="tl-metric-label">False</div>'
            f'<div class="tl-metric-value" style="color:#e74c3c">{false_count}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    for i, (claim, verification, live_context) in enumerate(results, 1):
        with st.container(border=True):
            st.markdown(f"### Claim {i} · `{claim.claim_type}`")
            st.markdown(f"> *{claim.original_quote}*")

            render_verdict_banner(verification.verdict, verification.explanation)

            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.markdown(
                    f'<div class="tl-metric-card">'
                    f'<div class="tl-metric-label">Confidence</div>'
                    f'<div class="tl-metric-value">{verification.confidence_score}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with metric_cols[1]:
                st.markdown(
                    f'<div class="tl-metric-card">'
                    f'<div class="tl-metric-label">Source Credibility</div>'
                    f'<div class="tl-metric-value">{verification.source_credibility_score}/100</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="tl-fact-box"><strong>Current Real Fact</strong><br/>'
                f'{verification.current_real_fact}</div>',
                unsafe_allow_html=True,
            )

            with st.expander("Live web sources used"):
                st.text(live_context)
                f'{verification.current_real_fact}</div>',
                unsafe_allow_html=True,
            )

            with st.expander("Live web sources used"):
                st.text(live_context)
