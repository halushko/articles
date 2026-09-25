# Access Recovery Support Policy

- **Document owner:** Service Operations
- **Policy reference:** OPS-AR-001
- **Audience:** Service Desk, Identity Operations, Platform Operations
- **Review cycle:** Annual or after a material change to the sign-in architecture

## Purpose

This policy defines how support teams handle reports from users who cannot
access the corporate software product. Its purpose is to provide a consistent
response regardless of whether the underlying problem belongs to the user's
account, the authentication service or the software platform.

This policy applies from the first contact with the user until the support
record is closed or responsibility is transferred through a formal
escalation. Requests for new access, role changes and planned maintenance are
outside its scope.

## Operating principles

Every access problem must have an incident record in the service management
system. Chat messages and phone calls may be used to collect information, but
they do not replace the incident record.

The Service Desk remains responsible for communication with the user. A
specialist team may diagnose and restore the affected component, but the
specialist does not close the incident on behalf of the Service Desk.

Recovery work should follow the least disruptive option available. Analysts
must preserve diagnostic evidence before changing an account, authentication
policy or running service. Emergency actions that bypass normal change
controls require approval from the incident manager.

## Service priorities

Priority is based on business impact and urgency rather than on the order in
which requests arrive.

| Priority | Typical situation | Initial response target |
|---|---|---|
| Critical | Most users cannot sign in, or a customer-facing production service is unavailable | 15 minutes |
| High | A business-critical team is blocked and no workaround exists | 30 minutes |
| Normal | One or several users are affected and a workaround may be available | 4 business hours |
| Low | Access is degraded but normal work can continue | 1 business day |

The incident manager must be informed immediately when the available evidence
suggests a widespread outage or a security event.

## Responsibilities

The Service Desk records the request, assesses its priority and decides which
specialist team should investigate it. The decision should be based on
observable symptoms, not on a guess about the technical root cause.

Identity Operations investigates account state, authentication policy and
federated sign-in. Platform Operations investigates service availability,
application health and failed dependencies. Each team records its findings and
recovery action in the existing incident.

After technical work is complete, the Service Desk performs an independent
check of the result. A successful technical command or a completed deployment
does not by itself prove that the user can access the product.

## Expected handling flow

When a report is received, the Service Desk creates or updates an incident,
records the affected user and service, assesses the impact and classifies the
request. Evidence of account lockout, repeated authentication rejection or an
MFA problem normally leads to Identity Operations. Failed health checks,
elevated error rates or reports from several unrelated users normally lead to
Platform Operations.

The assigned specialist follows the relevant recovery runbook and returns the
result, supporting evidence and any remaining risk to the Service Desk. The
Service Desk then verifies access and decides whether the incident has been
resolved. If normal access has not been restored, an escalation must be
registered before the user is given the final update.

The user must be told what was restored, whether any workaround remains in
place and where further updates will be published. The incident may be closed
only after the verification result, communication and closure reason have been
recorded.

## Related procedures

- *Service Desk Guide for Access Incidents* describes intake, classification
  and specialist hand-off.
- *Identity and Authentication Recovery Runbook* describes account and
  federated sign-in recovery.
- *Software Platform Recovery Runbook* describes application and dependency
  recovery.
- *Recovery Verification, Escalation and Closure Standard* defines the final
  verification and closure requirements.
