# Neural_Network_Charity_Analysis

## Introduction and Overview

AlphabetSoup is a philantropic foundation dedicated to helping organizations that protect the environment, improve people's well being and unify the world. The company has donated over $10 billion over the past 20 years. This money has been used to invest in life saving technologies and organize reforestation groups around the world. As a data scientist we want to analyze the impact of each donation to ensure the organizations money is being used effectively. However, not every donation is impactful, some organizations may take the money and disappear. Therefore, the organization wants to understand which organizations are worth donating to and which are high risk. Therefore, the organization wants a data driven solution that can help us accurately predict whether a donation yields a positive result or not. We believe that the problem is too complex for traditional supervised learning methods such as linear regression, random forest or even decision trees. Therefore, we want to design a neurel network to solve this problem. Neurel networks are powerful machine learning techniques modeled after neurons in the brain. Therefore, we will design a deep learning neurel network with the help of the python tensorflow library in order to see how well we can predict whether a donation is high risk or low risk with the help of data collected by the organization.


## Data Sources and Description

For this analysis we have a csv containing 34000 organizations that have received funding for Alphanet Soup over the years. Our list of variables include identifications columns, application type, affiliated sector of industry of the applicant, the government classification for the organization, use case for funding, organization type, active status, income classification, special considerations, and the funding amount requested and lastly whether the money was used effectively i.e. whether the donation yielded success. 

## Results 
* Found IS_SUCCESSFUL to be the target variable as it is a binary classifier predicting whether the donation was successful or not
* In initial deep learning model used all the features except identification columns, uses relu activation and two hidden layers and received a predictive accuracy of 72.6% percent and loss of 0.5538
* After exploring the data found that SPECIAL_CONSIDERATIONS and STATUS variables were categorically imbalanced and therefore were noisy and removed from the features
* Second model reduces noisy features and neurons in hidden layers and yielded accuracy of 72.09% and model loss of 0.5658
* Third model uses 4 hidden layers and changes activation function to sigmoid yielding accuracy of 72.29% and a model loss of 0.5619
* Final model drastically increases bins for application type and government organization classification, uses four hidden layers of a much higher number of neurons with an activation function of relu yielding a 73.67 accuracy, highest among all models. 


## Data Preprocessing 

In this analysis our target is whether the donation yielded success which is the IS_SUCCESSFUL. This is a binary outcome variable helping us understand if the donation given to the organization by AlphabetSoup yielded a positive outcome according to AlphabetSoup's performance critera. Variables considered features for the model are application type, affiliated sector of industry, government classification for organization, use case for funding, organization type, income classification and funding amount requested. After running the initial model with all variables, we decided to look at a few variables using the grouby and count function to optimize feature selection.

![alt text](https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/images/Preprocessing_1.PNG)
![alt text](https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/images/Preprocessing_2.PNG)

We found that the special considerations and the status variables were very imbalanced in their categories, with only 27 values of yes in special considerations and 5 values of 0 (inactive) in status. Therefore, we conclude that these variables only add noise to the data and will not bring any positive contribution to the model due to categorical imbalance in the data as we have too many values of one category and too few of another. Therefore, in our second model (AlphanetSoupCharity_Optimization_1) we decided to leave out SPECIAL_CONSIDERATIONS and STATUS as these are noisy features in our data.

## Compiling, Training and Evaluating Models 

### Model 1: Initial Deep Neurel Net 
link:https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb

In our initial neurel net deep learning model, we use all the features in the model except for identification columns. We use a relu activation function with 100 epochs, two hidden layers of 24 and 12 respectively and output layer with sigmoid function, we end up with a total of 1,441 parameters, plenty of parameters of help us find patterns in the data. Our model yields an accuracy of 72.6% with a model loss of 0.5539.

#### Model 1 Performance Results 
![alt text](https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/images/ASPC_Model_1_Performance.PNG)

In the graphs we see that the loss metrics reduces downward and stays constant after 80 epochs. On the other hand the accuracy score stays constant after 40 epochs with a huge rise after 5 epochs, a gradual rise from 5 epochs to 40 and a wobbly line showing how the line roughly remains constant after 40 epochs. 

#### Model 1 Graphs for Accuracy and Loss
![alt text](https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/images/ASPC_Model_1_graphs.PNG)

Overall this model performs well for a first model but we would like accuracy up to at least 75 to 80% so but we would like to experiment with the data, exploring the data to see which features are creating noise, and whether changing the bins, activation functions, hidden layers and neurons has any impact on the model. 

### Model 2: Removing Noise and Fewer Neurons
link: https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/AlphanetSoupCharity_Optimization_1.ipynb

In our second neurel net deep learning model, we go a step further in our initial model and remove SPECIAL_CONSIDERATIONS and STATUS variables. As mentioned above, these observations had an imbalance in terms of observations in these categories. We use a relu activation function with 50 epochs, reduced from 100 in the initial model and two hidden layers of 8 and 5, reduced from 24 and 12, and our output layer with a sigmoid function, we end up with a total of 395 parameters, less than the previous model, allowing us to see whether there is a drastic increase or decrease in performance as a result of this change. Our model yields an accuracy of 72.09% with a model loss of 0.5657.

#### Model 2 Performance Results 
![alt text](https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/images/ASPC_Model_2_Performance.PNG)

The graph shows model loss drastically reduce till the 10th epoch, after which we see an overall gradual reduction till around the 40th epoch, after which model loss is pretty constant. Accuracy rises drastically till the 10th epoch and stays more or less constant till the 50th epoch. 

#### Model 2 Graphs for Accuracy and Loss
![alt text](https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/images/ASPC_Model_2_graphs.PNG)

This model performs slightly worse than our initial model (accuracy of 72.6%), however, there is little difference in performance of the two models. We find that reducing number of neurons or epochs has little impact on performance. Therefore, it may be reasonable to assume that we should try to use a different activation functions or add hidden layers to see if this makes a difference. 

### Model 3: Adding Layers and Changing Activation Function
link:https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization_2.ipynb

In our third attempt, we preprocess similar to the second model, removing noise from the model. However, after seeing that reducing the number of neurons barely affected the model, we try to add more hidden layers, increasing hidden layers up to 4 with 8, 5, 4, and 3 neurons in the 4 layers, and a sigmoid output layer, resulting in a total of 455 parameters. We also use the sigmoid activation function instead of relu to see if changing the activation function affects model performance. We end up with an accuracy of 72.29% and a model loss of 0.5618

#### Model 3 Performance Results  
![alt text](https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/images/ASPC_Model_3_Performance.PNG)

We see a drastic reduction in model loss up to the 10th epoch after which we see a gradual reduction up to the 40th epoch, after which model loss stays constant. As for accuracy, we see a drastic increase up to the 4th or 5th epoch, staying the same afterwards. Therefore, we probably did not need 50 epochs in this model. 

#### Model 3 Graphs for Accuracy and Loss
![alt text](https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/images/ASPC_Model_3_graphs.PNG)

In conclusion, this model also performs around the same level as the other models, performing very slightly better than the second model but not better than our initial model, only yeilding an accuracy of 72.29, much below our required 75% minimum threshold. Adding a different activation function and even adding more hidden layers did not have any effect on the model.

### Final Optimized Model: More Neurons, Relu Activation and our Best Result
link: https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization_Final%20.ipynb

In our final attempt, after experimenting with the last few models, we decide to try one last time to optimize the model. We increase our bins for application types (11 bins) and government organization classification (12 bins), hoping these additional bins will help us uncover new trends. We change the activation function back to relu and maintain our four layers and we add a lot more neurons to a model since this combination is probably the only one we have not tried yet along with the fact that we need more neurons for the additional bins we have created. Thus, we find ourselves with 4 layers of 50, 24, 12 and 8 neurons respectively with a sigmoid output layer. We end up with our best answer yet, but only by around an additional percentage point compared to the rest, we end up with a 73.67% accuracy and a 0.5444 model loss. This is our best result yet by only by a little. 

![alt text](https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/images/ASPC_Model_4_Performance.PNG)

As for the graphs, we find that the model loss drops and stays constant after the 40th epoch, similar to other models. Our accuracy similarly sees a steep rise initially up to the 6th-7th epoch and then rises in a volatile manner, staying somewhat constant after the 40th epoch. 

![alt text](https://github.com/mobinapiracha/Neural_Network_Charity_Analysis/blob/main/images/ASPC_Model_4_graphs.PNG)

### Summary 
In conclusion, none of the models are successful in helping us classify successful donations to unsuccessful donations. Overall, our final model performs best out of all models with an accuracy of 73.67% (compared to 72.6, 72.09, and 72.29), however, all models fail to reach up to the 75% minimum threshold. If after four attempts and many different combinations we are unable to reach a decent accuracy score, this is an indication that either our features just don't help us classify the data very well or that we may want to use other models. The problem with neurel networks is that they involve a lot of trial and error because they don't allow much interpretability. This inability to see which features performed well leaves a lot of ambiguity in understanding model performance. Therefore, we may want to use logistic regression or even random forest classifiers as these models would allow us to measure the impact of different features on the outcome, allowing us to better understand feature importance may help us understand the next steps to improve the performance of the model.  
