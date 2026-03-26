# DS 4320 Project 1: Detecting Bot Accounts on Twitter

**Name:** Conor Gibbons

**NetID:** hjd3db

### Executive Summary

...

## Problem Definition

**Initial Problem:**
Detecting online bots

**Refined Problem:**
Can a binary classifier accurately distinguish between bot and human 
Twitter accounts based on account metadata and tweet behavior patterns 
such as follower count, posting frequency, and hashtag usage?

### Refinement Rationale
This refinement narrows the scope of the initial problem to focus on a single domain, Twitter, where bot activity is well documented and 
labeled data is publicly available. By constraining the problem to Twitter specifically, we can define the feature space to account metadata and tweet behavior patterns such as follower count, posting frequency, and hashtag usage which are consistent and measurable across all accounts. This sets the problem up as a binary classification task with a clear target. This helps avoid the issues that would arise from trying to generalize bot detection across multiple platforms with different structures.

### Project Motivation
The project was motivated by the growing concern that bots present to social media. In the age of AI, bots on social media will become harder to detect as their patterns more accurately mimic humans. These bots can cause widespread damage for a platform including spreading misinformation, 
manipulating trends, and amplifying political messages at a large scale. The ability to automatically detect bot accounts is therefore critical for platforms to mitigate the influence of automated accounts on public opinion. Manual detection is impossible at such a large scale, so a machine learning classifier trained on behavior and account signals offers a scalable solution.

### Press Release Headline

...

## Domain Exposition

### Terminology:
| Term | Category | Definition |
|------|----------|------------|
| Social Bot | Core Concept | An automated account on social media designed to mimic human behavior and interact with real users |
| Bot Label | Core Concept | A binary indicator denoting whether an account is a bot (1) or human (0) |
| Binary Classification | ML Task | A supervised learning task where the model predicts one of two possible labels, in this case bot or human |
| Account Metadata | Core Concept | Structural information about a Twitter account such as follower count, verified status, and creation date |
| Behavioral Signal | Core Concept | A measurable pattern derived from account activity such as posting frequency, retweet rate, or hashtag usage |
| Follower Count | Account Feature | The number of other accounts following a given user; bots often have abnormally low or inflated counts |
| Verified Status | Account Feature | A boolean indicator of whether Twitter has officially verified an account as authentic |
| Accuracy | Performance KPI | The proportion of correctly classified accounts out of all accounts |
| Precision | Performance KPI | Of all accounts predicted as bots, the proportion that actually are bots |
| Recall | Performance KPI | Of all actual bot accounts, the proportion the model correctly identified |
| F1 Score | Performance KPI | The harmonic mean of precision and recall; useful given potential class imbalance between bots and humans |
| Training Set | ML Concept | The portion of the dataset used to fit the classifier |
| Test Set | ML Concept | The held out portion of the dataset used to evaluate performance on unseen examples |

### Domain
This project lives in the domains of social media, cybersecurity, and machine learning. Twitter (now X) is one of the largest public discourse platforms in the world with real-time conversation across politics, news, entertainment, and finance. This openness makes it highly vulnerable to manipulation by automated accounts which can operate at scale. Bot detection is an active area of research with platforms investing heavily in automated moderation systems to identify and remove bot accounts. The machine learning approach to this problem treats detection as a supervised classification task, where models learn to distinguish bots from humans using patterns in account behavior and metadata. This domain requires attention to class imbalance, since the proportion of bots to humans in any dataset may not reflect real-world distributions, as well as to fairness since false positives risk flagging true human accounts.

### Background Reading
| Title | Description | Link |
|-------|-------------|------|
| Online Human-Bot Interactions: Detection, Estimation, and Characterization (Varol et al., 2017) | The foundational paper in Twitter bot detection. Introduces the BotOrNot framework, estimates 9-15% of active Twitter accounts are bots, and establishes the feature categories used in nearly all subsequent work | https://arxiv.org/pdf/1703.03107 |
| Twitter Bot Detection Using Diverse Content Features and Applying Machine Learning Algorithms | Academic paper covering feature-based ML approaches to bot detection including account metadata and content features | https://www.mdpi.com/2071-1050/15/8/6662 |
| Improving Social Bot Detection Through Aid and Training | Study on human ability to detect bots, good motivation for why automated detection matters | https://pmc.ncbi.nlm.nih.gov/articles/PMC11382440/ |
| Integrating Higher-Order Relations for Enhanced Twitter Bot Detection | Research on using behavioral signals like retweets and hashtags for detection, directly relevant to your features | https://link.springer.com/article/10.1007/s13278-024-01372-0 |
| BotArtist: Generic Approach for Bot Detection via Semi-Automatic Methods | Covers lightweight feature-based detection without requiring heavy API access, closest to your approach | https://arxiv.org/pdf/2306.00037 |

## Data Creation

### Provenance
The raw data for this project was sourced from Kaggle. The dataset, titled 
"Twitter Bot Detection Dataset", was downloaded as a single CSV file named 
`bot_detection_dataset.csv`. It contains 11 features capturing both account-level 
metadata and tweet-level behavior signals for a collection of Twitter accounts, 
each labeled as either a bot or a human. The dataset was collected from the 
Twitter platform and includes attributes such as follower count, verified status, 
tweet content, retweet count, mention count, hashtags, location, and account 
creation timestamp.

The raw CSV was ingested into a Jupyter notebook where it was processed into a 
fully normalized relational dataset consisting of four tables: `users`, `tweets`, 
`hashtags`, and `locations`. No external APIs were used in the data acquisition 
process. The original CSV is preserved in the `data/raw/` directory of this 
repository as the unmodified source of record, and all transformations applied 
to produce the final tables are documented and reproducible via the pipeline 
notebook.