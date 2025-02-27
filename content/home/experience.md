---
# An instance of the Experience widget.
# Documentation: https://wowchemy.com/docs/page-builder/
widget: experience

# This file represents a page section.
headless: true

# Order that this section appears on the page.
weight: 40

title: Experience
subtitle:

# Date format for experience
#   Refer to https://wowchemy.com/docs/customization/#date-format
date_format: Jan 2006

# Experiences.
#   Add/remove as many `experience` items below as you like.
#   Required fields are `title`, `company`, and `date_start`.
#   Leave `date_end` empty if it's your current employer.
#   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
experience:
  - title: Software Engineer / Machine Learning Engineer
    company: Adobe
    company_url: 'https://www.adobe.com/'
    company_logo: 
    location: San Jose, California
    date_start: '2023-02-06'
    date_end: ''
    description: |2-
        * Making the entire model training simpler one PR at a time by improving the training infrastructure and reducing the time to start training and evaluating a model checkpoint. I have actively been involved in development and PR reviews for changes that are used for production trainings where it is important to continuously determine the quality of image generations. 
        * Created a new model pipeline in the production repo that brings identity preservation into Firefly's image offerings. This was an entirely new experience as the parity between research and production pipelines is often not same in the beginning. 
        * Synthetic data generation, subject to data laws, for the next generation of Firefly Image model. This entails experimenting with a lot of models, to see how the generated data impacts the model because having better data almost always leads to a bump in quality. 
        * Deploying various models for rapid internal prototyping to see how good the models are, or how can they be used by end users, by using them on a user interface. 

  - title: Software Developer
    company: The Luminosity Lab
    company_url: 'https://theluminositylab.com/'
    company_logo: 
    location: Tempe, Arizona
    date_start: '2021-03-01'
    date_end: '2022-12-12'
    description: |2-
        * Developed an Android app for a company that aims to reduce stress through wearables. 
        * Improved the credit rewards system to increase the mobile app usage for Bank of the West.
        * Simulating the state of Arizona for analysing the effects of business decisions based on past data. 
        * Connected small businesses with student volunteers by developing a symbiotic web application. 
        * UI developer for a web application that lets users post and explore business ideas. 

  - title: Software Engineer Intern
    company: Adobe Sensei
    company_url: 'https://www.adobe.com/sensei.html'
    company_logo: 
    location: Remote
    date_start: '2022-05-16'
    date_end: '2022-08-05'
    description: |2-
        * Revamped the entire prompt UI for providing better customization options to the user for generating images.
        * Added seed functionality so that users can get different images for the same prompts. 
        * Integrated image search for generating creative variations of the searched image. 
        
  - title: Full Stack Developer
    company: Pirimid Fintech
    company_url: 'https://pirimidtech.com/'
    company_logo: 
    location: Ahmedabad, India
    date_start: '2020-08-20'
    date_end: '2020-12-15'
    description: |2-
        * Eliminated the need for in-person identity verification between banks and individual users by developing an e-KYC platform. Also optimized the code to reduce multiple API calls to a single API call.
        * Migrated stock positions data dashboards to Grafana for improving the efficiency of data filtering.
        
  - title: Open Source Contributor
    company: Habitica
    company_url: 'https://habitica.com/'
    company_logo: 
    location: Remote
    date_start: '2020-06-05'
    date_end: '2020-08-10'
    description: |2-
        * Fixed the re-casting of 2 major skills, Stealth and Chilling Frost, which impacted all the users. 
        * Fixed redundant invites and member count bugs for parties, and helped new contributors to set up the code base.
  
  - title: Software Intern
    company: MAQ Software
    company_url: 'https://maqsoftware.com/'
    company_logo: 
    location: Mumbai, India
    date_start: '2020-01-06'
    date_end: '2020-05-29'
    description: |2-
        * Worked on the frontend and backend changes for a 24x7 live platform which receives a high volume of transactions (>100,000) operated by a large US-based MNC. 

design:
  columns: '2'
---
