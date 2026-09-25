# Checkout Process

As a customer, I want to pay for an order, so that I can complete the checkout.

## Payment Flow

The user opens the checkout page. The system loads available payment methods, e.g. card, PayPal, etc. The user submits the form and then the system validates the input.

If payment fails, the system shows an error message. After that, the user can try again.

## Acceptance Criteria

- Given the cart contains items
- When the user submits a valid payment form
- Then the order is created and then confirmation email is sent

## API

POST /api/v1/orders/{orderId}/payments - Creates a payment for the selected order. Returns v1.2 response format.
GET /api/v1/orders/{orderId} - Returns order details.
