# Neural_Network_Charity_Analysis

## Introduction 

AlphabetSoup is a philantropic foundation dedicating to helping organizations that protect the environment, improve people's well being and unify the world. The company has donated over $10 billion over the past 20 years. This money has been used to invest in life saving technologies and organize reforestation groups around the world. As a data scientist we want to analyze the impact of each donation to ensure the organizations money is being used effectively. However, not every donation is impactful, some organizations may take the money and disappear. Therefore, the organization wants to understand which organizations are worth donating to and which are high risk. Therefore, the organization wants a data driven solution that can help us accurately predict whether a donation yields a positive result or not. We believe that the problem is too complex for traditional supervised learning methods such as linear regression, random forest or even decision trees. Therefore, we want to design a neurel network to solve this problem. Neurel networks are powerful machine learning techniques modeled after neurons in the brain. Therefore, we will design a deep learning neurel network woth the help of the python tensorflow library. 

## Data Sources and Description

For this analysis we have a csv containing 34000 organizations that have received funding for Alphanet Soup over the years. Our list of variables include identifications columns, application type, affiliated sector of industry of the applicant, the government classification for the organization, use case for funding, organization type, active status, income classification, special considerations, and the funding amount requested and lastly whether the money was used effectively i.e. whether the donation yielded success. 

## Data Preprocessing 

In this analysis our target is whether the donation yielded success. This is a binary outcome variable helping us understand if the donation given to the organization by AlphabetSoup yielded a positive outcome according to AlphabetSoup's performance critera. Variables considered features for the model are application type, affiliated sector of industry, government classification for organization, use case for funding, organization type, income classification and funding amount requested. After running the initial model with all variables, we decided to look a look at a few variables using the grouby and count function.

![alt text](https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/images/Preprocessing_1.PNG)
![alt text](https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/images/Preprocessing_2.PNG)





