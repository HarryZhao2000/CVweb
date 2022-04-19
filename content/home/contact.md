---
# An instance of the Contact widget.
widget: contact

# This file represents a page section.
headless: true

# Order that this section appears on the page.
weight: 130

title: Contact
subtitle:

content:
  # Automatically link email and phone or display as text?
  autolink: true
  
  # Email form provider
  form:
    provider: netlify
    formspree:
      id:
    netlify:
      # Enable CAPTCHA challenge to reduce spam?
      captcha: false

  # Contact details (edit or remove options as required)
  email: harry.zhao@drexel.edu
  phone: 445 208 8781
  address:
    street: 3737 Lancaster Avenue
    city: Philadelphia
    region: PA
    postcode: '19104'
    country: United States
    country_code: US

design:
  columns: '2'
---
