# Service Activation and Line Test Runbook

- **Runbook reference:** HIF-ACT-008
- **Runbook owner:** Network Activation Operations
- **Primary tools:** Provisioning Orchestrator, Access Network Manager, Radius Platform
- **Entry condition:** Field installation has returned an installed result

## Verify the installation hand-off

Review the reserved port, access technology, gateway identifier and technician
measurements. Compare the returned identifiers with the order and network
inventory before provisioning the service.

## Provision the access service

Create the subscriber service in the Provisioning Orchestrator with the agreed
speed profile and customer contract identifier. Bind the service to the
reserved network port and assigned gateway. Store the generated circuit and
subscriber identifiers in the order.

## Run network tests

Read the physical link state and signal measurements in the Access Network
Manager. Confirm that the gateway receives the expected configuration and can
establish an authenticated session through the Radius Platform.

Run a connectivity check to the provider edge and record the negotiated access
profile. Do not report success only because the provisioning command completed.

## Decide the activation outcome

Treat the activation as active when the physical link is stable, subscriber
authentication succeeds and the ordered profile is applied.

Treat the activation as failed when the link remains down, authentication is
rejected or the provisioned profile does not match the order.

## Handle a failed activation

Preserve the failed command and test identifiers. Reverse only the incomplete
provisioning objects that prevent a safe retry. Create a technical fault for the
team that owns the failed layer and keep the customer order out of billing.

## Return the activation result

Update the order with the circuit identifier, active profile, test evidence and
activation status. Return either an active result or a failed result with the
fault reference and next responsible team.
