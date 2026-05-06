"""
Column name mappings for datasets
"""


# Column Names for- Customer Care Emails (rtweera/customer_care_emails)
CCE_SOURCE_TEXT_COL = "message_body"
CCE_SOURCE_TICKET_COL = "thread_id"
CCE_SOURCE_ESCALATION_COL = "email_criticality"
CCE_SOURCE_SENTIMENT_COL = "customer_satisfaction"
CCE_ESCALATION_HIGH = "high"


# Column Names for- Customer Support Tickets (ajverse/customer-support-tickets-crm-dataset)

CST_SOURCE_TICKET_COL = "Ticket_ID"
CST_SOURCE_TEXT_COL = "Ticket_Description"
CST_SOURCE_SUBJECT_COL = "Ticket_Subject"
CST_SOURCE_PRIORITY_COL = "Priority_Level"
CST_SOURCE_SATISFACTION_COL = "Satisfaction_Score"
CST_PRIORITY_HIGH = "high"
CST_PRIORITY_MEDIUM = "medium"


# Column Names for - Synthetic IT Tickets (KameronB/synthetic-it-callcenter-tickets)
SIT_SOURCE_TICKET_COL = "number"
SIT_SOURCE_TEXT_COL = "content"
SIT_SOURCE_REASSIGNED_COL = "reassigned_count"
SIT_SOURCE_SENTIMENT_COL = "info_score_close_notes"

# Standardized Column Names (Output Format)
STANDARD_TICKET_ID_COL = "ticket_id"
STANDARD_TEXT_COL = "conversation_text"
STANDARD_ESCALATED_COL = "escalated"
STANDARD_SENTIMENT_COL = "sentiment_score"
STANDARD_CONVERSATION_TEXT_COL = "conversation_text"
STANDARD_PROCESSED_TEXT_COL = "processed_text"
