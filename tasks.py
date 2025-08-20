from crewai import Task

def create_fetch_task(agent):
    return Task(
        description="Fetch new messages from the inbox and extract details including support tokens.",
        agent=agent,
        expected_output="List of new messages with sender, subject, body, date, and support token"
    )

def create_classify_task(agent, subject, body):
    return Task(
        description=f"Classify email with subject: {subject} and body: {body}",
        agent=agent,
        expected_output="JSON with team, category, confidence, reasoning"
    )

def create_response_task(agent, request_details):
    return Task(
        description=f"Generate response for request: {request_details}",
        agent=agent,
        expected_output="Professional response text including support token"
    )