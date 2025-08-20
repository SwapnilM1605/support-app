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
    # Combine content for searching
    search_content = f"{subject} {body}"
    
    # Multiple patterns to catch tokens in different formats
    patterns = [
        r'SUPPORT-[A-Z0-9]{8}',                    # Exact format: SUPPORT-A1B2C3D4
        r'Reference[:\s]*([A-Z0-9-]{13,15})',      # "Reference: SUPPORT-A1B2C3D4"
        r'Ticket[:\s]*([A-Z0-9-]{13,15})',         # "Ticket: SUPPORT-A1B2C3D4"
        r'#[:\s]*([A-Z0-9-]{13,15})',              # "#SUPPORT-A1B2C3D4"
        r'Support[:\s]*([A-Z0-9-]{13,15})',        # "Support: SUPPORT-A1B2C3D4"
        r'ID[:\s]*([A-Z0-9-]{13,15})',             # "ID: SUPPORT-A1B2C3D4"
        r'Case[:\s]*([A-Z0-9-]{13,15})',           # "Case: SUPPORT-A1B2C3D4"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, search_content, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]  # Get the first group if it's a tuple
            if match.startswith('SUPPORT-') and len(match) == 13:  # SUPPORT- + 8 chars
                return match
            elif 'SUPPORT-' in match.upper():
                # Extract just the SUPPORT- part
                support_match = re.search(r'SUPPORT-[A-Z0-9]{8}', match, re.IGNORECASE)
                if support_match:
                    return support_match.group(0)
    
    # Check in common reply patterns
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

def agent_classify_and_assign(agent, subject, body):
    """Classifies an email and assigns it to the most appropriate team and category."""
    # Check if this is a reply by looking for thread token first
    thread_token = extract_thread_token(body, subject)
    if thread_token:
        return None, None, 0.0, f"Reply detected with token {thread_token} - classification will be inherited from thread"
    
    # Fallback to subject-based reply detection
    is_reply = subject.lower().startswith(('re:', 'fw:')) or any(
        phrase in body.lower() for phrase in [
            'on wrote:', 'on sent:', 'original message', 'forwarded message'
        ]
    )
    
    if is_reply:
        return None, None, 0.0, "Reply detected - classification will be inherited from thread"
    
    # Rest of the original classification logic...
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

    team_context = """## Available Support Teams ##
Each team has specific responsibilities:
"""
    for team_name, data in team_groups.items():
        team_entry = f"""### {team_name} Team ###
Description: {data['description']}
"""
        team_context += team_entry

    team_prompt = f"""
You are an expert support email classifier. Your task is to analyze the email and assign the best team based on their descriptions.

Follow this step-by-step process in your reasoning:
1. Summarize the main concern of the email in 1-2 sentences.
2. Evaluate each team: For each available team, explain if and how the email's concern matches the team description. Be specific about matching keywords or concepts. If no match, explicitly state why the team is not suitable.
3. Select the best team: Choose the team with the strongest match, explaining why. If no team matches well, strictly use Internal and explain why no other team fits.

### Classification Guidelines ###
- Use ONLY the team names provided exactly as listed
- Match the email's main concern to team descriptions closely. Do not force a match if it's weak or based on generic terms (e.g., common words like 'report' or 'review' do not justify a match unless context aligns perfectly).
- IT team handles ONLY technical issues like networks, hardware, software, account access, passwords. It does NOT handle business or administrative matters.
- HR handles ONLY employee-related issues like leave, payroll, benefits. It does NOT handle business operations or reporting.
- For topics outside defined team scopes, strictly use Internal.
- Only use 'Internal' team when no other team is appropriate, prioritizing it over incorrect assignments.
- Avoid overgeneralizing terms to force a match.

{team_context}

### Email to Classify ###
Subject: "{subject}"
Body: "{body}"

### Required Output Format ###
{{
  "team": "exact_team_name_from_list",
  "confidence": 0.0-1.0,
  "reasoning": "Detailed step-by-step explanation as per the process above"
}}

Example Output for unmatched email:
{{
  "team": "Internal",
  "confidence": 0.8,
  "reasoning": "1. Email discusses a business report. 2. IT does not match as it handles technical issues, not business matters. HR does not match as it handles employee-related issues, not reports. Internal is default. 3. Best is Internal as no specific team covers business reports."
}}
"""
    try:
        team_task = Task(
            description=team_prompt,
            expected_output="Valid JSON with team, confidence and reasoning",
            agent=agent
        )
        crew = Crew(agents=[agent], tasks=[team_task], verbose=0)
        raw_team_response = crew.kickoff()
        
        json_match = re.search(r"\{.*\}", raw_team_response.strip(), re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON in team response")
        
        parsed_team = json.loads(json_match.group(0))
        team_name = parsed_team.get("team", "").strip()
        team_confidence = float(parsed_team.get("confidence", 0.0))
        team_reasoning = parsed_team.get("reasoning", "").strip()
        
        if team_name not in team_groups:
            team_name = "Internal"
            team_confidence = 0.0
            team_reasoning += "\nInvalid team selected, falling back to Internal"
        
        if team_name == "Internal":
            category_name = "Other"
            cat_confidence = team_confidence
            cat_reasoning = "Default to Other for Internal team as no specific category applies"
        else:
            cat_data = team_groups[team_name]
            cat_context = f"""## Categories for {team_name} Team ##
Description: {cat_data['description']}
Categories:
"""
            for cat in cat_data["categories"]:
                cat_context += f"- {cat['name']}: {cat['description']}\n"
            
            cat_prompt = f"""
You are an expert support email classifier. Given the selected team {team_name}, assign the best category based on the descriptions.

Follow this step-by-step process in your reasoning:
1. Summarize the main concern of the email in 1-2 sentences.
2. Evaluate each category: For each category, explain if and how the email's concern matches the category description. Be specific about keywords and context.
3. Select the best category: Choose the category with the strongest match, explaining why. If no category matches well, strictly use 'Other' and explain why no specific category fits.

### Classification Guidelines ###
- Use ONLY the category names provided exactly as listed
- Prefer specific categories over 'Other' only if there's a strong, direct match to the category description
- Do not force matches based on generic terms (e.g., common words like 'report' do not imply a specific category)
- For IT, 'Password reset' applies only to explicit password-related issues
- For topics outside defined categories, use 'Other'

{cat_context}

### Email to Classify ###
Subject: "{subject}"
Body: "{body}"

### Required Output Format ###
{{
  "category": "exact_category_name_or_Other",
  "confidence": 0.0-1.0,
  "reasoning": "Detailed step-by-step explanation as per the process above"
}}

Example Output for unmatched category:
{{
  "category": "Other",
  "confidence": 0.7,
  "reasoning": "1. Request about a business report. 2. Password reset does not match as it covers password issues, not business matters. Other is default. 3. Best is Other as no specific category matches."
}}
"""
            cat_task = Task(
                description=cat_prompt,
                expected_output="Valid JSON with category, confidence and reasoning",
                agent=agent
            )
            crew = Crew(agents=[agent], tasks=[cat_task], verbose=0)
            raw_cat_response = crew.kickoff()
            
            json_match = re.search(r"\{.*\}", raw_cat_response.strip(), re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON in category response")
            
            parsed_cat = json.loads(json_match.group(0))
            category_name = parsed_cat.get("category", "").strip()
            cat_confidence = float(parsed_cat.get("confidence", 0.0))
            cat_reasoning = parsed_cat.get("reasoning", "").strip()
            
            valid_cats = [c["name"] for c in cat_data["categories"]]
            if category_name not in valid_cats:
                category_name = "Other"
                cat_confidence = 0.0
                cat_reasoning += "\nInvalid category, falling back to Other"
        
        overall_confidence = min(team_confidence, cat_confidence)
        overall_reasoning = f"Team Selection:\n{team_reasoning}\n\nCategory Selection:\n{cat_reasoning}"
        
        validator_agent = create_validator_agent(agent.llm)
        cat_desc = next((c['description'] for c in team_groups[team_name]['categories'] if c['name'] == category_name), "No description")
        validation_prompt = f"""
Validate if the assigned team '{team_name}' and category '{category_name}' logically match the email.

Team description: {team_groups[team_name]['description']}
Category description: {cat_desc}
Email subject: "{subject}"
Email body: "{body}"

Check:
- Does the email content directly and specifically relate to the team description? Generic terms like 'report' or 'review' do not justify a match unless context is clear.
- Does the category description explicitly cover the email's main concern?
- If the email involves topics not covered by the assigned team's description or category, it must be Internal/Other.
- Reject weak or forced matches with a clear explanation.

Output ONLY JSON:
{{
  "valid": true or false,
  "reason": "Explanation why valid or not"
}}

Example for unmatched email:
{{
  "valid": false,
  "reason": "Business report email assigned to IT/Password reset, but IT handles technical issues like passwords, not business matters. Should be Internal/Other."
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
                return "Other", "Internal", 0.0, overall_reasoning + f"\nValidation failed: {parsed_val.get('reason', '')}"
        
        return category_name, team_name, overall_confidence, overall_reasoning

    except Exception as e:
        fallback_category, fallback_team = keyword_based_classify(f"{subject} {body}")
        return (
            fallback_category,
            fallback_team,
            0.0,
            f"Fallback to keyword classification due to error: {str(e)}"
        )

def keyword_based_classify(text):
    # Check if this appears to be a reply
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
            team_scores[team] = 0
        
        # Team name (highest weight)
        if team.lower() in text_low:
            team_scores[team] += 10
        
        # Team description (moderate weight, focus on specific terms)
        if tc.team_description:
            words = set(w for w in tc.team_description.lower().split() if len(w) > 4)
            for word in words:
                if word in text_low:
                    team_scores[team] += 3
        
        # Category name (high weight)
        if tc.category_name.lower() in text_low:
            team_scores[team] += 8
        
        # Category description (moderate weight, focus on specific terms)
        if tc.category_description:
            words = set(w for w in tc.category_description.lower().split() if len(w) > 4)
            for word in words:
                if word in text_low:
                    team_scores[team] += 3
    
    if team_scores:
        best_team = max(team_scores, key=team_scores.get)
        best_score = team_scores[best_team]
        if best_score < 5:  # High threshold to avoid weak matches
            return "Other", "Internal"
    else:
        return "Other", "Internal"
    
    cats = [tc for tc in team_categories if tc.team_name == best_team]
    cat_scores = {}
    for cat in cats:
        score = 0
        if cat.category_name.lower() in text_low:
            score += 10
        if cat.category_description:
            words = set(w for w in cat.category_description.lower().split() if len(w) > 4)
            for word in words:
                if word in text_low:
                    score += 3
        cat_scores[cat.category_name] = score
    
    if cat_scores:
        best_cat = max(cat_scores, key=cat_scores.get)
        if cat_scores[best_cat] > 5:  # High threshold for category
            return best_cat, best_team
    
    return "Other", best_team