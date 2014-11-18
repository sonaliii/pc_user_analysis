PC Usage Trends
================

**App usage and retention analysis for PC photo-sharing**

Introduction:
The purpose of this program is to identify which app users are most likely to invite **more** users to the app. Using their publicly available and shared data from Facebook, this code creates a feature matrix and a random forest to identify the most valuable users. 

In addition, users' comments and captions are used to identify the most common usage trends for the app. A TF-IDF vectorizer and non-negative matrix factorization are used for topic modeling.

Getting Started:
- comments.py: imports comment text and groups by user
- circle_comments.py: imports comment text and groups by circle (album)
- captions.py: imports photo caption text and groups by user
- referrals.py: imports user referral data (who invited whom to the app)
- fb.py: imports facebook data; combines with text from captions and comments; random forest
- fb_virality.py: predicts user virality using various features; random forest
- fb_retention.py: predicts user retention using various features; logistic regression