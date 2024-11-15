# Project Overview 

This project aims to elevate the performance of our India-based e-commerce platform through data-driven insights. With products sourced globally, we encounter distinct challenges in understanding customer behaviour, optimising inventory, and refining pricing strategies. By tackling these challenges, we aim to enhance operational efficiency, boost customer satisfaction, and increase overall profitability.
								
# Contributors
Richelle - Data Analyst

Royston - Data Analyst

Shirlyn - Data Analyst

Yi Faye - Data Analyst

Covan Seah - Data Scientist

Zhi Chao - Data Scientist

Arnav - Data Scientist 

Samuel - Data Scientist

# Setup instructions 
1. Clone the repository 
 git clone https://github.com/isaroyston/3101project.git
 cd 3101project

2. Create virtual environment 
python3 -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'


3. Install dependencies 
pip install -r requirements.txt
pip install seaborn 
pip install plotly
pip install streamlit


4. Download the required CSVs here: https://drive.google.com/open?id=10rMN8liJnhmSy9-4mPLp9SRtQOMCpxW-

   Place CSVs at the same level as the root folder				

# Usage Guidelines 
Each folder in our git repository corresponds to a specific question. The following is the list of the folders and their associated content:

**Bonus1/**

- bonus1.ipynb : This file contains the implementation of a predictive model for identifying potential vip customers

**Bonus2/** 

- bonus2.ipynb : This file contains the implementation of Natural Language Processing  to analyse customer reviews and feedback

**Bonus3/** 

- bonus3.ipynb : This file contains the implementation of an AI-driven product recommendation system to increase cross-selling and upselling

**Bonus4/** 
- bonus4.ipynb : This file contains the implementation of a Personalized Shopping Assistant LLM

**config/**
- db.yaml : This file contains the credentials to connect to the remote database hosted with Amazon RDS

**dashboard/**

- campaign_sales_analysis.py : This file contains the code to run the streamlit dashboard for monitoring key e-commerce metrics as part of the bonus question. 

- functions.py : This file contains the reusable python functions that are imported and called by campaign_sales_analysis.py and main_page.py for the dashboard.

- main_page.py : This file contains the code to run the streamlit dashboard which was displayed during the roadshow.

**database/**

- __ init __.py

- api_service.py: This file defines the GET endpoint that retrieves all rows from the remote database

- database.py: This file contains the code to connect to the remote database using the credentials in ../config/db.yaml

- models.py: This file defines the column names and types for our table in the database

- schema.py: This file defines the format for rows returned from the database, ensuring consistency in data types

**Subgroup A Q1/**

- behavioural_analysis.ipynb: This file contains the in-depth analysis of behaviours of different customer segments

- behavioural_analysis.py: functions folder 

- historical_analysis.ipynb: This file contains the analysis of historical trends of customer purchases

- Magic_revenue_generator.ipynb: This file contains a calculator for marketing returns based on yearly ROI

- rfm_analysis_kmeans.ipynb: This file contains pipeline for customer segmentation (kmeans)

- rfm_analysis_manual.ipynb: This file contains pipeline for customer segmentation (rules bases)

- Rfm_analysis.py: This file contains functions for rfm analysis

**Subgroup A Q2/**

- q2.ipynb : This file contains the visualisations for analysis of customer retention and lifetime value.

- q2.py : This file contains the functions that are imported and called by q2.ipynb.

**Subgroup A Q3/**

- q3.ipynb : This file contains the visualisations for analysis of effectiveness of marketing channels and campaigns.

- q3.py : This file contains the functions that are imported and called by q3.ipynb.

**Subgroup_B_Q1**

- dsa3101Bq1.ipynb : This file contains the implementation for optimising inventory levels

**Subgroup_B_Q2**

- code.ipynb: This file contains the code for generating the dynamic pricing model to answer this question

**Subgroup_B_Q3/**

- __ init __.py

- models/: This folder contains the serialised Prophet models for each supplier. Each .json file corresponds to a Prophet model for a specific supplier that has been fitted with historical data

- api_service.py: This file defines the POST endpoint to retrieve average delivery time forecasts for a particular supplier

- q3.ipynb: Contains code for EDA and Prophet model evaluation for Subgroup B Q3

- train.py: This file contains the code to fetch all entries from the hosted database, filtering for each supplier, fitting them on a Prophet model and finally serialising them to be saved in ./models
