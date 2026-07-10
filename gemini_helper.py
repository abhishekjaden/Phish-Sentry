# Silence two cosmetic upstream warnings from the google-genai SDK's pydantic
# schema handling during function-calling. They do not affect behavior.
import warnings as _warnings
_warnings.filterwarnings("ignore", message=r".*is not a Python type.*")
_warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings")

import json
import random
from datetime import datetime
import os
from google import genai
from google.genai import types

import requests as _agent_requests

# --- Agent tool configuration (native Gemini function-calling) ---
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:9000")
RAG_URL = os.environ.get("RAG_URL", "http://localhost:9100")
_AGENT_TIMEOUT = 30


def classify_url(url: str) -> dict:
    """Classify a URL as phishing or legitimate using the PhishSentry URL model.

    Returns the calibrated phishing probability plus signal overlays: whether the
    domain is a homograph/IDN lookalike, the resolved final URL and redirect chain,
    and a fresh_domain flag for very recently registered domains.

    Args:
        url: The URL or link to analyze.
    """
    try:
        r = _agent_requests.post(f"{INFERENCE_URL}/predict/url", json={"url": url}, timeout=_AGENT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"url classifier unavailable: {e}"}


def classify_text(text: str) -> dict:
    """Classify an email or message body as phishing or legitimate using the
    PhishSentry RoBERTa text model.

    Returns is_phishing, a confidence percentage, and the phishing probability.

    Args:
        text: The email or message body to analyze.
    """
    try:
        r = _agent_requests.post(f"{INFERENCE_URL}/predict/text", json={"text": text}, timeout=_AGENT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"text classifier unavailable: {e}"}


def search_threat_intel(query: str) -> list:
    """Search the PhishSentry threat-intelligence knowledge base for context on
    phishing techniques, indicators, and mitigations relevant to the query.

    Use this to ground an explanation of WHY an observed signal (homograph,
    fresh domain, redirect, urgency lure, etc.) indicates phishing. Returns the
    most relevant reranked passages with their titles.

    Args:
        query: A short description of the phishing topic or indicator to look up.
    """
    try:
        r = _agent_requests.post(f"{RAG_URL}/retrieve", json={"query": query, "top_k": 4}, timeout=_AGENT_TIMEOUT)
        r.raise_for_status()
        return [{"title": c.get("title"), "content": c.get("content")} for c in r.json()]
    except Exception as e:
        return [{"error": f"knowledge base unavailable: {e}"}]


_AGENT_SYSTEM = (
    "You are PhishSentry's phishing-analysis agent. Given a URL or an email/message, "
    "investigate it with your tools and return a clear verdict.\n\n"
    "Tool guidance:\n"
    "- If the input is a URL or link, call classify_url.\n"
    "- If the input is an email or message body, call classify_text.\n"
    "- Use search_threat_intel to ground your explanation in the knowledge base "
    "(e.g. why a homograph, fresh domain, redirect, or urgency lure is dangerous).\n\n"
    "When reading classify_url output, note that resolved.blocked=true means the "
    "resolver could not safely fetch the link (unreachable or SSRF-guarded), NOT that "
    "it was flagged as malicious; do not cite \"blocked\" as evidence of phishing.\n\n"
    "Then respond with: (1) a verdict - Phishing, Suspicious, or Likely legitimate; "
    "(2) the key evidence the tools returned (calibrated probability, homograph/IDN, "
    "redirects, domain age, suspicious language); and (3) a brief grounded explanation. "
    "Be concise and practical. Never invent signals the tools did not return."
)


class PhishingQuiz:
    def __init__(self):
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Add it to your .env file or export it in your shell."
            )
        self.client = genai.Client(api_key=api_key)

    def generate_quiz_questions(self, difficulty="medium"):
        prompt = f"""
        Generate 5 multiple choice questions about phishing awareness for cybersecurity education.
        Difficulty level: {difficulty}
        
        Each question should:
        1. Test practical phishing identification skills
        2. Have 4 options (A, B, C, D)
        3. Have only one correct answer
        4. Include real-world scenarios
        5. Cover different phishing types (email, URL, social engineering, etc.)
        
        Return ONLY a valid JSON array with this exact structure:
        [
            {{
                "id": 1,
                "question": "Question text here",
                "options": {{
                    "A": "Option A text",
                    "B": "Option B text", 
                    "C": "Option C text",
                    "D": "Option D text"
                }},
                "correct_answer": "A",
                "explanation": "Why this answer is correct and others are wrong"
            }}
        ]
        
        Focus on practical scenarios like identifying suspicious emails, recognizing fake URLs, 
        understanding social engineering tactics, and best security practices.
        """
        
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction="You are a cybersecurity expert. Return only valid JSON with no extra text, markdown, or code fences."
                ),
            )
            questions_json = response.text.strip()
            
            if questions_json.startswith('```json'):
                questions_json = questions_json[7:-3]
            elif questions_json.startswith('```'):
                questions_json = questions_json[3:-3]
            
            questions = json.loads(questions_json)
            return questions
        
        except Exception as e:
            print(f"Error generating quiz: {e}")
            return self.get_fallback_questions()
    
    def get_fallback_questions(self):
        return [
            {
                "id": 1,
                "question": "Which of the following is the most reliable way to verify if an email is legitimate?",
                "options": {
                    "A": "Check if it has company logos",
                    "B": "Verify the sender through a separate communication channel", 
                    "C": "Look for spelling mistakes",
                    "D": "Check if it asks for personal information"
                },
                "correct_answer": "B",
                "explanation": "Always verify through independent channels like calling the company directly."
            },
            {
                "id": 2,
                "question": "Which URL is most likely to be a phishing attempt?",
                "options": {
                    "A": "https://www.paypal.com/signin",
                    "B": "https://www.paypal-secure-login.com/signin",
                    "C": "https://paypal.com/help",
                    "D": "https://www.paypal.com/account"
                },
                "correct_answer": "B",
                "explanation": "Phishing URLs often add extra words like 'secure' or 'login' to a legitimate brand name to appear trustworthy while using a different domain."
            },
            {
                "id": 3,
                "question": "You receive an urgent email from your bank saying your account will be suspended unless you click a link immediately. What should you do?",
                "options": {
                    "A": "Click the link immediately to avoid suspension",
                    "B": "Forward the email to friends for advice",
                    "C": "Go directly to your bank's official website or call them",
                    "D": "Reply to the email asking for more details"
                },
                "correct_answer": "C",
                "explanation": "Urgency is a common phishing tactic. Always navigate directly to official websites or call official numbers instead of clicking links in emails."
            },
            {
                "id": 4,
                "question": "What does 'HTTPS' in a URL indicate?",
                "options": {
                    "A": "The website is definitely legitimate and safe",
                    "B": "The connection is encrypted, but the site could still be malicious",
                    "C": "The website has been verified by the government",
                    "D": "The website cannot contain malware"
                },
                "correct_answer": "B",
                "explanation": "HTTPS only means the connection is encrypted. Phishing sites can and do use HTTPS, so it does not guarantee a site is legitimate."
            },
            {
                "id": 5,
                "question": "Which of the following is a sign of a phishing email?",
                "options": {
                    "A": "It comes from a known contact's exact email address",
                    "B": "It addresses you by your full name",
                    "C": "It contains a generic greeting like 'Dear Customer'",
                    "D": "It was sent during business hours"
                },
                "correct_answer": "C",
                "explanation": "Phishing emails often use generic greetings because they are sent in bulk and the attacker does not know the recipient's name."
            }
        ]

    def calculate_quiz_score(self, user_answers, questions):
        score = 0
        feedback = []
        for i, question_data in enumerate(questions):
            question_id = question_data['id']
            correct_answer = question_data['correct_answer']
            user_answer = user_answers[i] if i < len(user_answers) else None

            is_correct = (user_answer == correct_answer)
            if is_correct:
                score += 1

            feedback.append({
                'question_id': question_id,
                'is_correct': is_correct,
                'correct_answer': correct_answer,
                'user_answer': user_answer,
                'explanation': question_data['explanation']
            })
        return score, feedback

    def get_quiz_feedback(self, score, total_questions, feedback_details):
        if score == total_questions:
            return "Excellent! You have a strong understanding of phishing."
        elif score >= total_questions * 0.6:
            return "Good job! You understand the basics, but there's room for improvement."
        else:
            return "Keep learning! Review phishing tactics to improve your awareness."

    def get_grounded_answer(self, question, contexts):
        """
        Answer a question grounded strictly in the supplied context chunks.
        `contexts` is a list of dicts with at least 'title' and 'content'.
        Returns {"answer": <text>}.
        """
        if not contexts:
            return {"answer": "I don't have any threat-intelligence context on that yet."}

        blocks = []
        for i, c in enumerate(contexts, start=1):
            title = c.get("title") or c.get("doc_id") or f"Source {i}"
            blocks.append(f"[{i}] {title}\n{c.get('content', '')}")
        context_text = "\n\n".join(blocks)

        prompt = (
            "Answer the user's question using ONLY the context below. "
            "Cite the sources you use with their bracketed numbers, e.g. [1], [2]. "
            "If the context does not contain the answer, say so plainly instead of guessing.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {question}"
        )

        try:
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
            ]
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a cybersecurity expert explaining phishing threats. "
                        "Ground every claim in the provided context and cite sources by their "
                        "bracketed number. Be clear, concise, and practical. Do not invent "
                        "facts beyond the context."
                    )
                ),
            )
            return {"answer": response.text}
        except Exception as e:
            print(f"Error getting grounded answer: {e}")
            return {
                "answer": "I'm experiencing technical difficulties. Please try again shortly.",
                "error": str(e),
            }

    def get_agent_analysis(self, user_input):
        """
        Agentic analysis: Gemini autonomously decides which tools to call
        (classify_url, classify_text, search_threat_intel), runs them via the
        SDK's automatic function-calling loop, and composes a grounded verdict.
        Returns {"verdict": <text>, "tools_used": [...]}.
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=_AGENT_SYSTEM,
                    tools=[classify_url, classify_text, search_threat_intel],
                ),
            )
            tools_used = []
            afc = getattr(response, "automatic_function_calling_history", None) or []
            for content in afc:
                for part in (getattr(content, "parts", None) or []):
                    fc = getattr(part, "function_call", None)
                    if fc and getattr(fc, "name", None):
                        tools_used.append(fc.name)
            seen, ordered = set(), []
            for t in tools_used:
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
            return {"verdict": response.text, "tools_used": ordered}
        except Exception as e:
            print(f"Error in agent analysis: {e}")
            return {
                "verdict": "The analysis agent encountered an error. Please try again.",
                "error": str(e),
            }

    def get_chatbot_response(self, message):
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=message),
                    ],
                ),
            ]
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a cybersecurity expert specializing in phishing detection and prevention. "
                        "Help users understand phishing threats, how to identify suspicious emails and URLs, "
                        "and best practices to stay safe online. Be clear, concise, and practical in your answers."
                    )
                ),
            )
            return {"response": response.text}
        except Exception as e:
            print(f"Error getting chatbot response: {e}")
            return {
                "response": "I'm experiencing technical difficulties. Please try again shortly.",
                "error": str(e)
            }