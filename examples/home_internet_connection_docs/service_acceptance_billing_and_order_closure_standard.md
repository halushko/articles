# Service Acceptance, Billing and Order Closure Standard

- **Standard reference:** HIF-CLO-005
- **Standard owner:** Residential Customer Operations
- **Used in:** Order Management Portal, Billing Platform
- **Entry condition:** Network Activation has returned an activation result

## Review completion evidence

Confirm that the order contains the contract identifier, field completion
record, assigned equipment, circuit identifier and activation test result.
Return an incomplete hand-off to its owner instead of reconstructing missing
evidence from informal messages.

## Verify the customer-visible service

Ask the customer to connect a device through the supplied gateway and open a
normal internet service. When the customer is unavailable, run the approved
remote service check and retain a customer contact task.

Record who performed the check, when it occurred and whether basic connectivity
was available. A remote line test supports the result but does not prove that
the customer's local connection is usable.

## Decide the order outcome

Treat the order as completed when activation is active, the acceptance check
succeeds and no commercial or installation blocker remains.

Treat the order as pending remediation when activation failed, the customer
check failed or required completion evidence is missing.

## Handle pending remediation

Keep billing stopped and create a linked remediation task for the owning team.
Tell the customer what remains unresolved and give the next review time. Retain
the original order as the parent record for the follow-up work.

## Start billing and close the order

Set the billing start date to the accepted service date and activate the agreed
recurring charge in the Billing Platform. Send the customer the service summary,
support contact and equipment-return obligations.

Mark the order complete after the billing reference, acceptance evidence and
customer communication have been recorded.
