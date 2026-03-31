# Can a Classifier Tell the Difference Between a Real Person and a Bot on X (Twitter)?

## Social Media is Flooded with Fake Accounts

Every time you read a trending topic, reply to a post, or follow a recommendation 
on Twitter, there is a chance you are interacting with a bot. These automated 
accounts are designed to look and behave like real people, and they are everywhere. 
Researchers estimate that between 10% and 15% of active Twitter accounts are bots, 
with some estimates being even higher. These accounts quietly shape what content 
gets amplified and what opinions appear popular.

## The Problem: Bots Are Manipulating Online Conversation at Scale

Bots on Twitter are not just a nuisance, they are an active threat to the 
integrity of public discourse. They have been used to spread misinformation, 
artificially inflate support for political candidates, promote scam products, 
and drown out legitimate voices. The challenge is that modern bots are 
sophisticated enough to mimic real human behavior, making them difficult for 
ordinary users to identify. Manual detection is impossible at scale, and 
platforms have struggled to keep up. The question this project asks is simple: 
can a machine learning model automatically and accurately identify bot accounts 
based purely on how they behave on the platform?

## The Solution: A Machine Learning Bot Detector Trained on Real Twitter Data

This project builds an automated bot detection system trained on thousands of 
real labeled Twitter accounts. By analyzing patterns in how accounts interact 
with content including how often they use hashtags, mention other users, and receive 
engagement, the model learns to distinguish bots from humans without ever 
reading a single tweet. In testing, the system correctly identified bot and 
human accounts with 92% accuracy, demonstrating that behavioral patterns alone 
are enough to flag suspicious accounts. This kind of tool could help platforms, 
researchers, and everyday users make more informed decisions about who they are 
interacting with online.

## Chart
![Bot Detection Results](/figures/model2_results.png)