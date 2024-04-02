# MACHINE MAINTENANCE ANALYSIS

Group 3 analyzed a synthetic dataset for determining machine failures. Six features were provided, while a multivariate target included failure types. The end result was that the feature "Tool Wear" had the biggest impact on machine failure, though various features contribute to different failure types. A company in this situation should develop a preventative maintenance plan to mitigate these issues in the future.

--------------------------------------------------------------------------------------
The team approached this analysis in the following manner:

1) *Data overview*
2) *Analyses and conclusions*
3) *Next steps*
--------------------------------------------------------------------------------------
The following sections break down the overall approach:
1. **Data Overview**
    - 10,000 rows x 10 columns
    - ***IDs***
        - *UID*
            - Unique identifier ranging from 1 to 10000
        - *ProductID*
            - Consisting of a letter L, M, or H for low (50% of all products), medium (30%), or high (20%) as product quality variants and a variant-specific serial number    
    - ***Features (X)***
        - *Type*
            - Consisting of a letter L, M, or H for low, medium, or high
        - *Air temperature [K]*
            - Generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
        - *Process temperature [K]*
            - Generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K
        - *Rotational speed [rpm]*
            - Calculated from powepower of 2860 W, overlaid with a normally distributed noise
        - *Torque [Nm]*
            - Torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values
        - *Tool wear [min]*
            - The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. and a 'machine failure' label that indicates, whether the machine has failed in this particular data point for any of the following failure modes are true
    - ***Targets (Y)***
        - *Target*
            - 0 (No Failure)
            - 1 (Failure)
        - *Failure Type*
            - Heat Dissipation Failure
            - Overstrain Failure
            - No Failure
            - Power Failure
            - Random Failures
            - Tool Wear Failure
2. **Analyses and Conclusions**
    - ***Data Cleaning and Manipulation***
        - Dropped ID columns (e.g., UID, ProductID)
        - Converted feature "Failure Type" into dummy variables
        - Split the data into training and test sets  
        - Scaled the data in the feature columns
        - Used one-hot encoding for feature "Type" (e.g., L, M, H)
    - ***Pipeline***
        - xxx
    - ***Models***
        - xxx
    - ***xxx***
        - xxx

- ### Accuracy Scores
    | **Model** | **As-Is Data** | **Oversampled Data** | **Undersampled Data** |
    | --- | --- | --- | --- |
    | Logistic Regression | xxx | xxx | xxx |
    | K-Nearest Neighbors | xxx | xxx | xxx |
    | Decision Tree | xxx | xxx | xxx |
    | Random Forest | xxx | xxx | xxx |
    | Extremely Random Trees | xxx | xxx | xxx |
    | Gradient Boosting | xxx | xxx | xxx |
    | AdaBoost | xxx | xxx | xxx |
    | Support Vector Machine | xxx | xxx | xxx |
    | Naive Bayes | xxx | xxx | xxx |
    | xxx | xxx | xxx | xxx |

3. **Next Steps**
    - Develop a preventative maintanence plan
    - xxx
    - xxx


## Resources
- Kaggle: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

--------------------------------------------------------------------------------------
