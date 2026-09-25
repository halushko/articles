# Service Desk Guide for Access Incidents

- **Procedure reference:** SD-PR-014
- **Procedure owner:** Service Desk Manager
- **Used in:** Corporate ITSM
- **Applies to:** Reports that an existing user cannot access the software product

## Before you begin

Use this guide for an unexpected loss of access. Do not use it to request a new
account, add permissions or schedule a service change. If there is evidence of
credential theft or unauthorized access, stop this procedure and invoke the
security-incident process.

## Receive and record the report

A report may arrive through the support portal, telephone, email or an approved
chat channel. Acknowledge the report and confirm how the user can be contacted
during the investigation.

Search the service management system for an existing open incident describing
the same user and symptom. Update that incident when it represents the same
event; otherwise create a new one. The record must include:

- the affected user and business unit;
- the product, environment and sign-in URL;
- the time of the most recent successful and failed sign-in attempts;
- the exact message shown to the user;
- whether other users are affected;
- a callback or messaging channel.

Do not copy passwords, one-time codes or session tokens into the incident.

## Assess impact and assign priority

Use the priority rules in the *Access Recovery Support Policy*. Confirm the
number of affected users, the business activity that is blocked and whether a
workaround exists. Increase the priority when the same symptom is reported by
unrelated users or when monitoring shows a production outage.

Record the reasoning behind the selected priority. If the available facts are
insufficient, use the lower confirmed impact and revise it when more evidence
becomes available.

## Classify the incident

Select the service category and the symptom that best describe what the user
observes. Do not classify the root cause before it has been investigated.

The following indicators help choose the initial assignment:

| Observed evidence | Initial assignment |
|---|---|
| The account is reported as locked, expired or disabled | Identity Operations |
| The password is accepted but MFA or a conditional-access rule rejects the session | Identity Operations |
| A federated sign-in page fails for one organization | Identity Operations |
| Health monitoring reports that the product endpoint is unavailable | Platform Operations |
| Several unrelated users receive server errors after signing in | Platform Operations |
| There is not enough evidence to distinguish the cause | Keep with Service Desk while basic checks are completed |

Attach screenshots or log references when they are available. Before assigning
the incident, state what has already been checked and what question the
specialist is expected to answer.

## Select the diagnostic route

When the evidence points to an account or authentication problem, assign the
incident to Identity Operations and request the checks in the *Identity and
Authentication Recovery Runbook*.

When the evidence points to a service or application problem, assign it to
Platform Operations and request the checks in the *Software Platform Recovery
Runbook*.

If a widespread outage and an individual account problem are both plausible,
first confirm the public service health. Do not reset or unlock multiple
accounts merely because the application cannot be reached.

## Receive the specialist result

The specialist must return the diagnostic conclusion, the action performed and
the evidence collected. Review the update before contacting the user. A note
such as “fixed” or “service restarted” is not sufficient.

Continue with the *Recovery Verification, Escalation and Closure Standard*.
The Service Desk owns the final access check, user communication and incident
closure even when another team performed the recovery.
