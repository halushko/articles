# Identity and Authentication Recovery Runbook

- **Runbook reference:** IAM-RB-007
- **Owner:** Identity Operations
- **Primary tools:** IAM administration console, authentication logs, External IdP status console
- **Entry condition:** The incident has been assigned with evidence of an account or authentication problem

## Preparation

Read the Service Desk notes and confirm the user identifier, affected product,
environment and time of the failed sign-in. If any of these values is missing,
request it before changing the account.

Open the audit trail for the user. Record the incident reference in every
administrative action so that the previous and new state can be reconstructed.

## Check the account

Review whether the account exists and is enabled. Check its lock state,
expiration date and recent failed sign-in events. Confirm that the account is
associated with the expected organization and environment.

An unexpected disabled or expired state should be compared with lifecycle and
security events before it is changed. If the account was intentionally
disabled, return the incident without restoring access and explain the reason
to the Service Desk.

## Review authentication policy

Inspect the rules that apply to the failed sign-in. Pay particular attention to
multi-factor authentication enrollment, group membership, conditional-access
rules and recent policy changes. Compare the rejected event with a successful
event for the same product when one is available.

If a local policy error clearly explains the failure, correct the approved
setting and continue with the recovery action. An external-provider check is
not required when the authentication attempt never left the local identity
service.

## Check federated sign-in when required

Use the external identity provider console when the sign-in is federated or
the local evidence does not explain the failure. Review provider availability,
federation metadata, certificate validity and the corresponding single
sign-on event.

When the external provider is degraded, use only an approved failover or
temporary-access procedure. Do not weaken authentication policy to work around
an unconfirmed external outage.

## Restore access

Apply the least disruptive action that addresses the confirmed cause. Typical
actions include unlocking the account, correcting group membership, requiring
new MFA enrollment, refreshing federation metadata or repeating a failed
directory synchronization.

Record the previous state, the action taken and the resulting state. Ask the
Service Desk to perform the final user-facing access check; a successful
administrative action is only a recovery attempt until the result has been
verified.

If the approved actions do not restore access, document what has been excluded
and return the incident as unresolved. Do not repeat the same account changes
without new evidence.

## Evidence to return

- the account state before and after the investigation;
- relevant authentication event identifiers and timestamps;
- the policy or provider finding;
- the recovery action and audit reference;
- any temporary control or remaining risk;
- a clear statement that access is ready for verification or remains unresolved.
