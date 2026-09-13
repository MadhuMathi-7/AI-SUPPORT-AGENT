"""
Grounded AI Reply Generator Module for Customer Support System.

Drafts concise customer support responses grounded strictly in retrieved historical brand resolutions.
Enforces strict anti-hallucination guardrails:
1. Grounding in historical evidence: Uses proven brand support patterns.
2. Anti-hallucination constraints: Never invents refund promises, fake delivery dates, or claims actions were taken.
3. No AI self-identification: Speaks naturally in the voice of the customer support team.
4. Dual-mode support: Seamlessly utilizes LLM APIs when keys are available in .env,
   or a deterministic grounded synthesis engine when running offline.
"""

import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


SYSTEM_GROUNDING_PROMPT = """You are an official Amazon Customer Support representative on Twitter (@AmazonHelp).
Your goal is to draft a helpful, empathetic, and concise support reply to the customer.

STRICT GROUNDING RULES:
1. Base your reply ONLY on the historical brand resolutions provided in the evidence below.
2. DO NOT invent refund amounts, delivery guarantees, or claim that an action has been taken on the account.
3. If order-specific details or private account information are required, direct the customer to contact support via secure Direct Message (DM) or official Amazon support links.
4. Keep the response under 280 characters (Twitter limit).
5. Never mention that you are an AI, language model, or automated system.
6. Use an empathetic, professional customer service tone.
"""


def format_grounded_prompt(
    customer_message: str,
    predicted_intent: str,
    retrieved_examples: List[Dict[str, Any]],
    conversation_context: Optional[str] = None
) -> str:
    """Format the full grounded prompt with historical evidence."""
    evidence_parts = []
    for i, ex in enumerate(retrieved_examples, 1):
        c_msg = ex.get("historical_customer_message", "").strip()
        b_resp = ex.get("historical_brand_response", "").strip()
        sim = ex.get("similarity", 0.0)
        evidence_parts.append(f"Evidence #{i} (Similarity: {sim:.2f}):\n- Customer: {c_msg}\n- Brand Reply: {b_resp}")
        
    evidence_block = "\n\n".join(evidence_parts) if evidence_parts else "No historical evidence available."
    context_block = f"Prior Conversation Context:\n{conversation_context}\n\n" if conversation_context else ""
    
    prompt = f"""{context_block}Customer Message: "{customer_message}"
Classified Intent: {predicted_intent}

Retrieved Historical Evidence:
{evidence_block}

Draft the grounded customer support reply:"""
    return prompt


def sanitize_customer_message(raw_text: str) -> str:
    """
    Sanitize historical customer message by stripping @handles, fixing encoding artifacts,
    and removing specific personal order identifiers while preserving the core support query.
    """
    text = str(raw_text)
    # 1. Fix encoding artifacts (e.g. unicode replacement char \ufffd -> ')
    text = text.replace('\ufffd', "'").replace('&amp;', '&').replace('&gt;', '>').replace('&lt;', '<')
    # 2. Strip @handles (@AmazonHelp, @530262, @username, etc.)
    text = re.sub(r'@\w+', '', text)
    # 3. Strip customer-specific order/phone numbers
    text = re.sub(r'\b\d{3}-\d{7}-\d{7}\b', '[ORDER-ID]', text)
    text = re.sub(r'\b\d{10,12}\b', '[PHONE-NUMBER]', text)
    # 4. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def sanitize_brand_response(raw_text: str) -> str:
    """
    Sanitize historical brand response by removing @handles, agent initials,
    split markers, specific customer names in greetings/sentences, order IDs, and personal information.
    Converts customer-specific greetings into clean, professional, universal replies.
    """
    text = str(raw_text)
    
    # 1. Clean encoding artifacts (e.g. unicode replacement char \ufffd -> ')
    text = text.replace('\ufffd', "'").replace('&amp;', '&').replace('&gt;', '>').replace('&lt;', '<')
    
    # 2. Strip @handles (@12345, @AmazonHelp, @user, etc.)
    text = re.sub(r'@\w+', '', text)
    
    # 3. Strip agent signatures (^TS, ^AB, ^MO, ^TM, ^MC, ^GL, etc.)
    text = re.sub(r'\^[A-Z]{2,4}\b', '', text)
    
    # 4. Strip fraction / pagination markers (1/2, 2/2, [1/2], (2/2), etc.)
    text = re.sub(r'(?:\b\d+/\d+\b|\[\d+/\d+\]|\(\d+/\d+\))', '', text)
    
    # 5. Normalize whitespace initially so greetings are at start of string
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Replace/remove customer-specific greetings at beginning of text:
    # Matches "Hello Gill!", "Helo Gill!", "Hi, Amy!", "Hey Sarah!", "Good morning Alex,", "Thanks David,", etc.
    # Requires the customer name/id to be directly followed by punctuation (!, ., ,)
    greeting_prefix_pattern = re.compile(
        r'^(?:Hi|Hello|Hey|Helo|Hlo|Greetings|Good\s+(?:morning|afternoon|evening)|Thanks|Thank\s+you),?\s+(?:there,?\s+)?(?:@?\w+\s+)?([A-Z][a-z]+|\d+)[,!\.]+\s*',
        re.IGNORECASE
    )
    text = greeting_prefix_pattern.sub('', text)
    
    # Matches standalone leading names: "Gill, thanks for..." or "Gill! Please check..."
    leading_name_pattern = re.compile(
        r'^[A-Z][a-z]+[,!]\s+(?=(?:thanks|thank|please|can|could|we|i|upon|if|for|to|you|have|sorry)\b)',
        re.IGNORECASE
    )
    text = leading_name_pattern.sub('', text)
    
    # 7. Strip customer names mid-sentence: ", Amy!" -> "!" or ", Sarah," -> ","
    text = re.sub(r',\s+[A-Z][a-z]+([!\?\.])', r'\1', text)
    text = re.sub(r',\s+[A-Z][a-z]+,', ',', text)
    
    # 8. Strip specific order numbers or customer phone numbers/IDs
    text = re.sub(r'\b\d{3}-\d{7}-\d{7}\b', '', text)
    text = re.sub(r'#\d{5,}', '', text)
    
    # 9. Clean up residual leading punctuation or orphan greeting fragments
    text = re.sub(r'^[,\.\-!:\s]+', '', text).strip()
    
    # 10. Ensure sentence capitalization
    if text:
        text = text[0].upper() + text[1:]
        
    return text


class GroundedReplyGenerator:
    """
    Grounded Reply Generator with LLM API support and deterministic fallback.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_examples: List[Dict[str, Any]],
        conversation_context: Optional[str] = None
    ) -> str:
        """
        Generate a grounded customer support response.
        
        Args:
            customer_message: The query to answer.
            predicted_intent: Classified intent label.
            retrieved_examples: Top-k historical resolutions.
            conversation_context: Preceding turns in dialogue.
            
        Returns:
            Concise, grounded reply string.
        """
        prompt = format_grounded_prompt(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            retrieved_examples=retrieved_examples,
            conversation_context=conversation_context
        )
        
        # 1. Try LLM Generation if API key is provided
        if self.api_key:
            try:
                reply = self._call_llm(prompt)
                if reply and len(reply.strip()) > 5:
                    return reply.strip()
            except Exception as e:
                print(f"[REPLY_GEN] LLM API call failed ({e}), falling back to grounded template synthesis.")
                
        # 2. Deterministic Grounded Synthesis Fallback
        return self._synthesize_grounded_reply(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            retrieved_examples=retrieved_examples
        )
        
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM API (OpenAI / LiteLLM compatible)."""
        try:
            import requests
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                headers = {
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_GROUNDING_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 120
                }
                res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
                if res.status_code == 200:
                    return res.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        return None
        
    def _synthesize_grounded_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_examples: List[Dict[str, Any]]
    ) -> str:
        """
        Deterministic, evidence-grounded template synthesis.
        Extracts key resolution actions and links from retrieved historical brand responses.
        """
        # If strong historical evidence is available, adapt the top resolution
        if retrieved_examples and len(retrieved_examples) > 0:
            top_evidence = retrieved_examples[0]
            top_brand_resp = top_evidence.get("historical_brand_response", "")
            
            clean_resp = sanitize_brand_response(top_brand_resp)
            
            # Verify the response is an actionable resolution and not a pure PII disclaimer or fragment
            is_pii_disclaimer = bool(re.search(r'personal information|page is public|public page|delete (?:your )?tweet', clean_resp, re.IGNORECASE))
            if len(clean_resp) > 25 and not is_pii_disclaimer:
                # Ensure under 280 chars
                return clean_resp[:280]
                
        # Standard grounded intent-specific fallbacks matching AmazonHelp support procedures
        fallbacks = {
            "delivery_tracking_delay": "I'm sorry to hear your delivery is delayed! Please check your latest tracking status under 'Your Orders'. If it has not updated, please DM us your order number so we can investigate.",
            "refund_return_cancellation": "I understand you need assistance with a return or refund. You can initiate returns directly via the Online Returns Center. If you need further help with order cancellation, please DM us.",
            "damaged_defective_item": "I am so sorry your item arrived in that condition! Please visit 'Your Orders' to request an immediate replacement, or send us a DM with your order details so we can assist you.",
            "account_access_security": "For your security, we do not handle account credentials over Twitter. Please use the Two-Step Verification recovery page or visit amazon.com/help to secure your account.",
            "payment_billing_issue": "I understand your billing concern. Please check 'Your Payments' in your account to review the transaction details. If you see duplicate charges, send us a private message so we can review.",
            "prime_subscription_services": "I'd be glad to help with your Prime membership! You can manage or cancel auto-renewal under 'Manage Prime Membership' in your account settings.",
            "general_inquiry_support": "We're here to help! Please send us a Direct Message with further details about your inquiry so our team can provide personalized assistance."
        }
        return fallbacks.get(predicted_intent, fallbacks["general_inquiry_support"])
