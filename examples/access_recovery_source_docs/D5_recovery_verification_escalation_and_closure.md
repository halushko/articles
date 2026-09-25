# Recovery Verification, Escalation and Closure Standard

- **Standard reference:** SD-ST-009
- **Owner:** Service Desk Manager
- **Audience:** Service Desk analysts, specialist teams and incident managers
- **Used after:** Identity or platform recovery work has been returned to the Service Desk

## Review the recovery evidence

Before contacting the user, confirm that the specialist update identifies the
diagnostic conclusion, action performed and resulting technical state. Return
the incident to the specialist when required evidence is missing; do not infer
success from a completed task or deployment status alone.

## Verify the user-visible result

Ask the user to repeat the normal sign-in path when they are available. For a
shared outage, use a controlled test account and confirm the product endpoint,
authentication flow and a basic authorized operation. Do not ask the user to
send a password or one-time code.

Record who performed the check, when it was performed and what result was
observed. Monitoring data may support the result, but it does not replace a
user-path check when only one account was affected.

## Decide the incident outcome

Treat the incident as recovered when the expected access path succeeds and no
blocking symptom remains. A temporary workaround must be recorded together
with its owner and expiry, even when it allows the incident to be resolved.

Treat the incident as unresolved when verification fails, the symptom returns
immediately or the responsible specialist cannot complete the required
recovery action.

## Escalate an unresolved incident

Create an escalation linked to the original incident. Include the business
impact, checks already completed, evidence collected, attempted recovery
actions and the team or vendor expected to continue the investigation.

For critical and high-priority incidents, notify the incident manager before
transferring ownership. Do not close the original record until the escalation
reference and the next responsible owner are visible to the Service Desk.

## Inform the user

Tell the user whether access has been restored or further investigation is
required. Summarize the outcome in non-technical language, identify any
workaround and provide the next update time when an escalation remains open.

Do not expose internal credentials, raw security logs or personal information
belonging to another user. Record the communication channel and timestamp in
the incident.

## Close the incident

Select a closure category that matches the confirmed outcome. Add the recovery
or escalation reference, verification evidence and user communication to the
record. Check that no temporary access or emergency change has been left
without an owner.

Close the incident after the user has been informed. If the user cannot be
reached, follow the standard contact-attempt policy and record each attempt
before closure.
