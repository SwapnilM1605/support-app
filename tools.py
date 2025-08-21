import os
import sys
import imaplib
import email
from email.header import decode_header
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
from crewai import LLM, Crew, Task
from config import AZURE_CONFIG, LLM_SETTINGS, IMAP, SMTP
import re
import json
from datetime import datetime
import certifi
from models import db, TeamCategory
from agents import create_validator_agent
import PyPDF2
from docx import Document

load_dotenv()

def fix_windows_encoding():
    if os.name == 'nt':
        try:
            os.system('chcp 65001 > nul 2>&1')
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass

def create_llm():
    """Create and configure the LLM with Azure OpenAI"""
    os.environ["AZURE_API_KEY"] = AZURE_CONFIG.get("api_key", "")
    os.environ["AZURE_API_BASE"] = AZURE_CONFIG.get("endpoint", "")
    os.environ["AZURE_API_VERSION"] = AZURE_CONFIG.get("api_version", "")
    os.environ["AZURE_AD_TOKEN"] = ""  # Clear any existing token
    
    model = f"azure/{AZURE_CONFIG.get('model')}"
    
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()
    
    return LLM(
        model=model,
        temperature=LLM_SETTINGS["temperature"],
        api_key=os.environ["AZURE_API_KEY"],
        api_base=os.environ["AZURE_API_BASE"],
        api_version=os.environ["AZURE_API_VERSION"]
    )

def test_llm_connection(llm):
    """Test the LLM connection"""
    try:
        print("Testing LLM connection...")
        test_response = llm.call("Hello, this is a test. Please respond with 'Connection successful.'")
        print(f"LLM Test Response: {test_response}")
        return True
    except Exception as e:
        print(f"LLM connection failed: {e}")
        return False

def send_email_via_smtp(to_email, subject, body, from_email=SMTP["email"]):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)
    try:
        server = smtplib.SMTP(SMTP["host"], SMTP["port"])
        server.starttls()
        server.login(SMTP["email"], SMTP["password"])
        server.send_message(msg)
        server.quit()
        return True, None
    except Exception as e:
        return False, str(e)

def parse_imap_message(msg_bytes):
    msg = email.message_from_bytes(msg_bytes)
    
    sender = msg.get("From", "")
    
    subject, encoding = decode_header(msg.get("Subject", ""))[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding or "utf-8", errors="ignore")
    
    # Check if this is a reply
    is_reply = subject.lower().startswith("re:") or "re:" in subject.lower()
    
    # Extract original subject for thread matching
    original_subject = subject.replace("Re:", "").replace("RE:", "").strip()
    
    date_str = msg.get("Date", "")
    try:
        email_date = email.utils.parsedate_to_datetime(date_str)
    except (TypeError, ValueError):
        email_date = datetime.utcnow()
    
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(msg.get_content_charset() or "utf-8", errors="ignore")
    
    return sender, subject, original_subject, body, email_date, is_reply

def fetch_unseen_emails():
    results = []
    mail = None
    try:
        mail = imaplib.IMAP4_SSL(IMAP["host"], IMAP["port"])
        mail.login(IMAP["email"], IMAP["password"])
        mail.select("inbox")

        status, messages = mail.search(None, "(UNSEEN)")
        if status != "OK":
            return results
            
        message_nums = messages[0].split()
        if not message_nums:
            return results

        for num in message_nums:
            mail.store(num, '+FLAGS', '\\Seen')

        for num in message_nums:
            try:
                res, data = mail.fetch(num, "(RFC822)")
                if res == "OK":
                    sender, subject, original_subject, body, email_date, is_reply = parse_imap_message(data[0][1])
                    results.append((sender, subject, original_subject, body, email_date, is_reply))
            except Exception as e:
                print(f"Error processing message {num}: {e}")
                continue

    except Exception as e:
        print(f"IMAP connection error: {e}")
    finally:
        try:
            if mail:
                mail.close()
                mail.logout()
        except:
            pass
    return results

def extract_thread_token(body, subject=""):
    """Flexibly extract thread token from email body and subject"""
    search_content = f"{subject} {body}"
    
    patterns = [
        r'SUPPORT-[A-Z0-9]{8}',
        r'Reference[:\s]*([A-Z0-9-]{13,15})',
        r'Ticket[:\s]*([A-Z0-9-]{13,15})',
        r'#[:\s]*([A-Z0-9-]{13,15})',
        r'Support[:\s]*([A-Z0-9-]{13,15})',
        r'ID[:\s]*([A-Z0-9-]{13,15})',
        r'Case[:\s]*([A-Z0-9-]{13,15})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, search_content, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            if match.startswith('SUPPORT-') and len(match) == 13:
                return match
            elif 'SUPPORT-' in match.upper():
                support_match = re.search(r'SUPPORT-[A-Z0-9]{8}', match, re.IGNORECASE)
                if support_match:
                    return support_match.group(0)
    
    reply_patterns = [
        r'On.*wrote:.*?(SUPPORT-[A-Z0-9]{8})',
        r'From:.*?Sent:.*?(SUPPORT-[A-Z0-9]{8})',
        r'Original Message.*?(SUPPORT-[A-Z0-9]{8})',
    ]
    
    for pattern in reply_patterns:
        match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
    
    return None

def extract_text_from_file(file_path):
    """Extract text content from supported file types."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext == '.pdf':
            text = ''
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + '\n'
            return text
        elif ext in ['.docx', '.doc']:
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        else:
            return ''
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ''

def agent_classify_and_assign(agent, subject, body):
    """Classifies an email and assigns it to the most appropriate team and category."""
    thread_token = extract_thread_token(body, subject)
    if thread_token:
        return None, None, 0.0, f"Reply detected with token {thread_token} - classification will be inherited from thread"
    
    is_reply = subject.lower().startswith(('re:', 'fw:')) or any(
        phrase in body.lower() for phrase in [
            'on wrote:', 'on sent:', 'original message', 'forwarded message'
        ]
    )
    
    if is_reply:
        return None, None, 0.0, "Reply detected - classification will be inherited from thread"
    
    team_categories = TeamCategory.query.all()
    
    team_groups = {}
    for tc in team_categories:
        if tc.team_name not in team_groups:
            team_groups[tc.team_name] = {
                "email": tc.team_email,
                "description": tc.team_description or "No description available",
                "categories": []
            }
        team_groups[tc.team_name]["categories"].append({
            "name": tc.category_name,
            "description": tc.category_description or "No description"
        })

    team_context = """## Available Support Teams and Categories ##
Each team has specific responsibilities and associated categories:
"""
    for team_name, data in team_groups.items():
        team_entry = f"""### {team_name} Team ###
Description: {data['description']}
Categories:
"""
        for cat in data["categories"]:
            team_entry += f"- {cat['name']}: {cat['description']}\n"
        team_context += team_entry

    prompt = f"""
You are an expert support email classifier. Your task is to analyze the email and assign the most appropriate team and category based on their descriptions from the database.

Follow this step-by-step process in your reasoning:
1. Summarize the main concern of the email in 1-2 sentences.
2. Evaluate each team: For each team, check if the email's concern matches the team description or any of its category descriptions. Be specific about matching keywords, phrases, or concepts. If no match, explicitly state why the team is not suitable.
3. Evaluate each category within the best team: For the selected team, check each category description for the best match. Be precise about keyword or concept matches.
4. Select the best team and category: Choose the team and category with the strongest match, explaining why. If no specific match, use 'Internal' team and 'Other' category, explaining why no other options fit.

### Classification Guidelines ###
- Use ONLY the team and category names as listed in the provided context.
- Match the email's concern closely to team and category descriptions. Do not force matches based on generic terms (e.g., 'report' or 'issue' alone are not sufficient).
- IT team handles ONLY technical issues (e.g., networks, hardware, software, account access, passwords). It does NOT handle financial or administrative matters.
- HR handles ONLY employee-related issues (e.g., leave, payroll queries, benefits, onboarding). It does NOT handle financial transactions or technical issues.
- Finance handles financial matters (e.g., salary payments, expenses, invoices, tax issues). Salary credit issues (e.g., non-credited, delayed, or incorrect salary) belong to Finance, not IT or HR unless explicitly about employee benefits.
- Use 'Internal/Other' only when no other team/category matches, and explain why.
- Consider synonyms and related terms, but prioritize exact or near-exact matches to descriptions.
- Avoid misclassifications by cross-checking against all teams/categories.

{team_context}

### Email to Classify ###
Subject: "{subject}"
Body: "{body}"

### Required Output Format ###
{{
  "team": "exact_team_name_from_list",
  "category": "exact_category_name_or_Other",
  "confidence": 0.0-1.0,
  "reasoning": "Detailed step-by-step explanation as per the process above"
}}

Example Output for unmatched email:
{{
  "team": "Internal",
  "category": "Other",
  "confidence": 0.8,
  "reasoning": "1. Email discusses a business report. 2. IT does not match (technical issues only). HR does not match (employee-related issues only). Finance does not match (financial transactions only). 3. No specific category applies. 4. Internal/Other is the best fit as no team/category matches."
}}

Example Output for salary credit issue:
{{
  "team": "Finance",
  "category": "Salary Credit",
  "confidence": 0.95,
  "reasoning": "1. Email complains about salary not credited. 2. IT does not match (technical issues only). HR partially matches (payroll) but Finance directly handles salary disbursement. Finance description mentions 'salary disbursement' and category 'Salary Credit' explicitly covers 'salary not credited'. 3. Salary Credit category matches due to 'salary not credited' phrase. Other Finance categories (e.g., Taxation Issue) do not apply. 4. Finance/Salary Credit is the best match."
}}
"""
    try:
        task = Task(
            description=prompt,
            expected_output="Valid JSON with team, category, confidence, and reasoning",
            agent=agent
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=0)
        raw_response = crew.kickoff()
        
        json_match = re.search(r"\{.*\}", raw_response.strip(), re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON in response")
        
        parsed = json.loads(json_match.group(0))
        team_name = parsed.get("team", "").strip()
        category_name = parsed.get("category", "").strip()
        confidence = float(parsed.get("confidence", 0.0))
        reasoning = parsed.get("reasoning", "").strip()
        
        # Validate team and category existence
        valid_teams = [t.team_name for t in team_categories]
        if team_name not in valid_teams:
            team_name = "Internal"
            category_name = "Other"
            confidence = 0.0
            reasoning += "\nInvalid team selected, falling back to Internal/Other"
        else:
            valid_cats = [c["name"] for c in team_groups[team_name]["categories"]]
            if category_name not in valid_cats:
                category_name = "Other"
                confidence = min(confidence, 0.5)
                reasoning += f"\nInvalid category for team {team_name}, falling back to Other"
        
        # Validation step
        validator_agent = create_validator_agent(agent.llm)
        cat_desc = next((c['description'] for c in team_groups[team_name]['categories'] if c['name'] == category_name), "No description")
        
        validation_prompt = f"""
Validate if the assigned team '{team_name}' and category '{category_name}' are the BEST logical match for the email, considering ALL available teams and categories.

{team_context}

Assigned Team description: {team_groups[team_name]['description']}
Assigned Category description: {cat_desc}
Email subject: "{subject}"
Email body: "{body}"

Follow this step-by-step process:
1. Summarize the email concern.
2. Check match to assigned: Does the email directly relate to the assigned team/category descriptions? Explain specific matches or mismatches.
3. Check alternatives: Evaluate if any other team/category matches better. Be specific about why or why not.
4. Final validation: If the assigned is the best match, valid=true. If another matches better or assigned doesn't match at all, valid=false and explain.

### Validation Guidelines ###
- Reject if assigned team/category doesn't explicitly cover the concern (e.g., salary credit to IT is invalid as IT is for technical issues, not financial).
- Prefer teams/categories with direct keyword or concept matches to descriptions.
- If multiple possible, choose the most specific; invalidate if wrong one selected.
- Examples of invalid: Salary/payment issues to IT/Password reset (should be Finance/Salary Credit). Technical access to Finance.

Output ONLY JSON:
{{
  "valid": true or false,
  "reason": "Detailed step-by-step explanation"
}}

Example for invalid salary to IT:
{{
  "valid": false,
  "reason": "1. Concern: Salary not credited. 2. IT/Password reset doesn't match; IT is for technical issues, not payments. 3. Finance/Salary Credit matches better with 'salary disbursement' and 'salary not credited'. 4. Invalid, should be Finance/Salary Credit."
}}

Example for valid:
{{
  "valid": true,
  "reason": "1. Concern: Forgot password. 2. Matches IT/Password reset directly. 3. No other team handles passwords. 4. Valid as best match."
}}
"""
        val_task = Task(
            description=validation_prompt,
            expected_output="Valid JSON with valid and reason",
            agent=validator_agent
        )
        crew = Crew(agents=[validator_agent], tasks=[val_task], verbose=0)
        raw_val_response = crew.kickoff()
        
        json_match = re.search(r"\{.*\}", raw_val_response.strip(), re.DOTALL)
        if json_match:
            parsed_val = json.loads(json_match.group(0))
            if not parsed_val.get("valid", True):
                return "Other", "Internal", 0.0, reasoning + f"\nValidation failed: {parsed_val.get('reason', '')}"
        
        return category_name, team_name, confidence, reasoning

    except Exception as e:
        fallback_category, fallback_team = keyword_based_classify(f"{subject} {body}")
        return (
            fallback_category,
            fallback_team,
            0.0,
            f"Fallback to keyword classification due to error: {str(e)}"
        )

def keyword_based_classify(text):
    if any(phrase in text.lower() for phrase in [
        'on wrote:', 'on sent:', 'original message', 'forwarded message'
    ]):
        return None, None
        
    text_low = text.lower()
    team_categories = TeamCategory.query.all()
    
    team_scores = {}
    for tc in team_categories:
        team = tc.team_name
        if team not in team_scores:
            team_scores[team] = {'score': 0, 'category_scores': {}}
        
        # Score based on team description
        if tc.team_description:
            words = set(w for w in tc.team_description.lower().split() if len(w) >= 3)
            for word in words:
                if word in text_low:
                    team_scores[team]['score'] += 2
        
        # Score based on category name
        if tc.category_name.lower() in text_low:
            if tc.category_name not in team_scores[team]['category_scores']:
                team_scores[team]['category_scores'][tc.category_name] = 0
            team_scores[team]['category_scores'][tc.category_name] += 8
        
        # Score based on category description
        if tc.category_description:
            desc_words = set(w for w in tc.category_description.lower().split() if len(w) >= 3)
            for word in desc_words:
                if word in text_low:
                    if tc.category_name not in team_scores[team]['category_scores']:
                        team_scores[team]['category_scores'][tc.category_name] = 0
                    team_scores[team]['category_scores'][tc.category_name] += 3
    
    if team_scores:
        best_team = max(team_scores, key=lambda x: team_scores[x]['score'])
        best_score = team_scores[best_team]['score']
        if best_score < 5:
            return "Other", "Internal"
    else:
        return "Other", "Internal"
    
    best_cat_scores = team_scores[best_team]['category_scores']
    if best_cat_scores:
        best_cat = max(best_cat_scores, key=best_cat_scores.get)
        if best_cat_scores[best_cat] > 5:
            return best_cat, best_team
    
    return "Other", best_team