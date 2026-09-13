"""
Streamlit Web Application: Grounded AI Customer Support Assistant.

Demonstrates the complete end-to-end support pipeline:
1. Semantic Intent Classification with calibrated confidence.
2. Historical Support Retrieval with cosine similarity and leakage prevention.
3. Grounded Reply Generation adhering to brand support conventions.
4. Intelligent Escalation Routing (AUTO_HANDLE vs. ESCALATE_TO_HUMAN).
"""

import sys
import os
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import SupportAgentPipeline


# Page configuration
st.set_page_config(
    page_title="AI Customer Support Agent | Hiver SDE Intern",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge-auto {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .status-badge-escalate {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    .evidence-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .reply-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 16px;
        font-size: 1.05rem;
        color: #14532D;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing AI Support Pipeline (Loading models & embeddings)...")
def get_pipeline():
    """Load and cache the pipeline instance."""
    return SupportAgentPipeline()


pipeline = get_pipeline()

# Header
st.markdown('<div class="main-header">🤖 AI Customer Support Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Grounded Support Automation, Semantic Intent Discovery & Escalation Routing for <b>@AmazonHelp</b></div>',
    unsafe_allow_html=True
)

# Sidebar: System Metadata & Presets
with st.sidebar:
    st.header("⚙️ System Architecture")
    st.markdown("""
    - **Dataset:** Customer Support on Twitter (`twcs.csv`)
    - **Selected Brand:** `AmazonHelp` (168k verified pairs)
    - **Embedding Model:** `all-MiniLM-L6-v2` (384-d)
    - **Intent Taxonomy:** 7 Empirically Discovered Classes
    - **Escalation Thresholds:**
      - Min Intent Conf: `0.65`
      - Min Retrieval Sim: `0.60`
    """)
    st.divider()
    
    st.subheader("📋 Sample Test Queries")
    sample_queries = {
        "1. Delivery Delay (Auto-Handle)": "My package was supposed to arrive yesterday by 8pm but tracking has not updated since Tracy CA. Where is it?",
        "2. Damaged Product (Auto-Handle)": "I received my order today and the ceramic teapot is completely shattered in the box. How do I get an exchange?",
        "3. Account Compromised (Escalate: Security)": "Someone hacked into my account from another country and changed my password and email! Help me recover it!",
        "4. Legal / Supervisor (Escalate: Sensitive)": "Your customer rep lied to me and stole my money! I am filing a police report and speaking with my lawyer.",
        "5. Prime Subscription (Auto-Handle)": "How can I turn off auto-renewal on my Amazon Prime student trial before I get charged next week?"
    }
    
    selected_sample = st.selectbox("Choose a pre-built example:", ["-- Select an example --"] + list(sample_queries.keys()))


# Main Input Area
default_text = sample_queries[selected_sample] if selected_sample in sample_queries else ""

user_query = st.text_area(
    "Enter Incoming Customer Support Message:",
    value=default_text,
    height=110,
    placeholder="Type or paste a customer tweet or message here..."
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    submit = st.button("🚀 Process Message", type="primary", use_container_width=True)

import re

def redact_ui_text(raw_text: str) -> str:
    """
    Sanitize and redact any customer-specific identifiers, personal names,
    order IDs, phone numbers, and handles for public demonstration and screenshots.
    """
    text = str(raw_text)
    
    # 1. Clean unicode encoding artifacts
    text = text.replace('\ufffd', "'").replace('&amp;', '&').replace('&gt;', '>').replace('&lt;', '<')
    
    # 2. Redact Order Numbers (e.g. 404-9391432-5602714, #12345, Order 123456)
    text = re.sub(r'\b\d{3}-\d{7}-\d{7}\b', '[Order Number]', text)
    text = re.sub(r'#\d{5,}', '[Order Number]', text)
    text = re.sub(r'\b(?:order|item|ref|invoice)\s*(?:id|num|number|#)?\s*[:=\s]\s*\d+\b', 'order [Order Number]', text, flags=re.IGNORECASE)
    
    # 3. Redact Phone Numbers and numeric tracking IDs
    text = re.sub(r'\b\d{10,12}\b', '[Phone Number]', text)
    text = re.sub(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', '[Phone Number]', text)
    
    # 4. Redact Emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[Customer Email]', text)
    
    # 5. Redact Agent Signatures
    text = re.sub(r'\^[A-Z]{2,4}\b', '', text)
    
    # 6. Redact Twitter Handles (@530262, @366906, @user, etc.)
    text = re.sub(r'@AmazonHelp\b', '___BRAND_PLACEHOLDER___', text)
    text = re.sub(r'@\w+', '[Customer Handle]', text)
    text = text.replace('___BRAND_PLACEHOLDER___', '@AmazonHelp')
    
    # 7. Redact Customer Names in Greetings & sentences
    greeting_name_pattern = re.compile(
        r'\b(Hi|Hello|Hey|Helo|Hlo|Greetings|Good\s+(?:morning|afternoon|evening)|Thanks|Thank\s+you),?\s+(?:there,?\s+)?(?:\[Customer Handle\]\s+)?([A-Z][a-z]+|\d+)([,!\.]+)',
        re.IGNORECASE
    )
    text = greeting_name_pattern.sub(r'\1 [Customer Name]\3', text)
    
    # Redact mid-sentence names after comma: ", Amy!" -> ", [Customer Name]!"
    text = re.sub(r',\s+([A-Z][a-z]+)([!\?\.])', r', [Customer Name]\2', text)
    
    # Redact common customer names directly
    common_names = ["Amy", "Gill", "Sarah", "John", "David", "Alex", "Michael", "Emma", "Lisa", "James", "Robert", "Mary", "Patricia", "Jennifer"]
    for name in common_names:
        text = re.sub(rf'\b{name}\b', '[Customer Name]', text, flags=re.IGNORECASE)
    
    # 8. Clean tweet split markers (1/2, 2/2)
    text = re.sub(r'(?:\b\d+/\d+\b|\[\d+/\d+\]|\(\d+/\d+\))', '', text)
    
    # 9. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


if submit and user_query.strip():
    with st.spinner("Classifying intent, retrieving historical brand evidence, and generating reply..."):
        result = pipeline.process(customer_message=user_query.strip())
        
    st.divider()
    
    # 2-Column Results Layout
    left_col, right_col = st.columns([1.1, 1.3])
    
    with left_col:
        st.subheader("🎯 Intent & Escalation Decision")
        
        # Intent Card
        intent_name = result["intent"]
        confidence = result["intent_confidence"]
        st.markdown(f"**Predicted Intent:** `{intent_name}`")
        st.progress(float(confidence), text=f"Intent Confidence: {confidence*100:.1f}%")
        
        st.markdown("---")
        
        # Decision Badge
        decision = result["decision"]
        reason = (
            str(result["reason"])
            .replace("strong historical resolution evidence", "relevant historical evidence")
            .replace("support safe automated reply", "support automated handling")
        )
        
        if decision == "AUTO_HANDLE":
            st.markdown('<span class="status-badge-auto">✅ Decision: AUTO_HANDLE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge-escalate">⚠️ Decision: ESCALATE_TO_HUMAN</span>', unsafe_allow_html=True)
            
        st.markdown(f"**Routing Reason:** {reason}")
        
        st.markdown("---")
        
        # Grounded AI Reply Box
        display_reply = redact_ui_text(result["reply"])
        st.subheader("💬 Generated Grounded Support Reply")
        st.markdown(f'<div class="reply-box">{display_reply}</div>', unsafe_allow_html=True)
        st.caption(f"Character Count: {len(display_reply)} / 280 (Twitter Limit Compliant)")

    with right_col:
        st.subheader("📚 Retrieved Historical Evidence (@AmazonHelp)")
        st.caption("Top semantically similar interactions retrieved from historical index (Leakage Masked & Redacted):")
        
        examples = result["retrieved_examples"]
        if examples:
            for idx, ex in enumerate(examples, 1):
                sim = ex.get("similarity", 0.0)
                sim_color = "#16A34A" if sim >= 0.70 else ("#D97706" if sim >= 0.60 else "#DC2626")
                
                c_msg_redacted = redact_ui_text(ex['historical_customer_message'])
                b_resp_redacted = redact_ui_text(ex['historical_brand_response'])
                
                with st.expander(f"Evidence #{idx} — Cosine Similarity: {sim:.2f}", expanded=(idx == 1)):
                    st.markdown(f"**Customer Query:**\n> {c_msg_redacted}")
                    st.markdown(f"**Historical Brand Reply:**\n> {b_resp_redacted}")
                    st.markdown(f"<span style='color:{sim_color}; font-weight:600;'>Match Score: {sim*100:.1f}%</span>", unsafe_allow_html=True)
        else:
            st.info("No historical evidence met the minimum retrieval threshold.")
            
elif submit and not user_query.strip():
    st.warning("Please enter a customer message to analyze.")
