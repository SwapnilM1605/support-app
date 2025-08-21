from crewai import Agent

def create_email_fetcher_agent(llm):
    """Agent role: fetch details from mailbox"""
    return Agent(
        role="Email Fetcher",
        goal="Fetch new emails from the support mailbox and extract details including support tokens.",
        backstory="You connect to the email server and extract sender, subject, body, date, and support token from new messages.",
        llm=llm,
        verbose=False
    )

def create_classifier_agent(llm):
    """Agent role: classify and assign team/category"""
    return Agent(
        role="Email Classifier",
        goal="Classify incoming support emails and assign the best suitable team and category based on dynamic teams and categories.",
        backstory="You analyze the email content step by step and assign the most appropriate team and category from available options, falling back to Internal team as last option if no fit.",
        llm=llm,
        verbose=False
    )

def create_responder_agent(llm):
    """Agent role: generate responses"""
    return Agent(
        role="Response Generator",
        goal="Generate professional and polite responses to support requests including the support token.",
        backstory="You craft concise, helpful responses addressing customer concerns with clear next steps, including the support ticket number. You use provided knowledge base documents to provide accurate responses.",
        llm=llm,
        verbose=False
    )

def create_validator_agent(llm):
    """Agent role: validate classification"""
    return Agent(
        role="Classification Validator",
        goal="Validate if the assigned team and category logically match the email content based on descriptions.",
        backstory="You critically check if the classification makes sense, ensuring no hallucinations or mismatches.",
        llm=llm,
        verbose=False
    )