# Software Platform Recovery Runbook

- **Runbook reference:** PLT-RB-012
- **Owner:** Platform Operations
- **Primary tools:** Monitoring, centralized logging, deployment orchestrator
- **Entry condition:** The incident has been assigned with evidence of a service or application failure

## Preparation and safeguards

Confirm the affected service, environment and approximate start time. Review
active incidents and recent deployments before taking action. Preserve the
initial health metrics and relevant logs because a restart or rollback may
remove useful evidence.

Production changes must be executed through the deployment orchestrator.
Direct changes to running instances are not an acceptable recovery method
unless the incident manager has approved an emergency action.

## Confirm service availability

Check the public and internal health endpoints, network route, recent error
rate and request latency. Compare the result from more than one location when
the report may be network-specific.

If the service is healthy, compare the incident symptom with authentication
failures before changing the platform. Return the incident to the Service Desk
when the available evidence points back to an individual access problem.

## Inspect the application

Review instance or replica health, resource saturation, application logs and
the most recent deployment or configuration change. Determine whether the
failure is local to the application or whether a required downstream service
is unavailable.

When a recent application change clearly explains the symptom, prepare the
appropriate recovery action. A separate dependency investigation may be
skipped only when the local cause is supported by logs or deployment evidence.

## Inspect dependencies when the cause is unclear

Check the database, message broker, identity service and external APIs used by
the affected request path. Correlate their health with the time of the user
reports. Do not restart a healthy dependency as a diagnostic experiment.

Record which dependency failed and whether the owning team has an active
incident. If another team must act first, retain ownership of coordination
until the application can be tested again.

## Restore the platform

Choose the smallest controlled action that addresses the confirmed cause. This
may be a rollback, restart, scale operation, configuration restoration or
dependency failover. Execute the action through the orchestrator and retain its
change or deployment identifier.

After the action, confirm that instances are stable and that health indicators
have returned to an acceptable range. Send the technical result, before-and-
after evidence and any remaining risk to the Service Desk. The Service Desk
must still verify the user-visible access path before declaring the incident
resolved.

If the service cannot be stabilized, return an explicit unresolved result and
identify the team or vendor required for escalation.

## Evidence to return

- health and error-rate observations before and after recovery;
- relevant log or trace references;
- the confirmed application or dependency finding;
- the orchestrator change or deployment identifier;
- temporary measures and rollback conditions;
- a clear statement that access is ready for verification or remains unresolved.
