import pandas as pd
import numpy as np


# Normalisation of the dataset.
"""

First, we splitted the features into 4 categories: 
- The critical features: Hypertension + heart diseases + age
- The average features: Avg clucose level + bmi + smoking status 
- The minor feature: Residence type
- The irrelevant features: gender, id, work type, ever married

We have discarded the 4th group and normalised each features.
Categorical features has been set with this dict:
`cleanup_nums = {"Residence_type":     {"Rural": 1, "Urban": 0},
                "smoking_status":     {"never smoked": 0, "Unknown": 0, "smokes": 1.5, "formerly smoked": 1.25 }}`
We droped each rows who are in the last category.

We set the proportion of each rows in the distance calcul.

The proportion is set by a variable by category of features:
Critical features : 0.8
Average features: 0.5
Minor features : 0.2
Irrelevant features: 0.0

    """
def process_data(data):
    cleanup_nums = {"Residence_type":     {"Rural": 1, "Urban": 0},
                "smoking_status":     {"never smoked": 0, "Unknown": 0, "smokes": 1.5, "formerly smoked": 1.25 }}
    df = df.replace(cleanup_nums)
    df = data.drop(['id','gender','ever_married','work_type','stroke'],axis=1)
    for v in df:
        x = df[v]
        norm = np.linalg.norm(x)
        normal_array = x/norm
        df[v] = normal_array
    return df

def calculate_dist(d1, d2):
    age_dist  = abs(float(d1[0]) - float(d2[0])) * 0.8
    hypertension_dist = abs(float(d1[1]) - float(d2[1])) * 0.8
    heart_disease_dist = abs(float(d1[2]) - float(d2[2])) * 0.8
    smoking_status_dist = abs(float(d1[6]) - float(d2[6])) * 0.8
    avg_glucose_level_dist = abs(float(d1[4]) - float(d2[4])) * 0.5
    bmi_dist = abs(float(d1[5]) - float(d2[5])) * 0.5
    residence_type_dist = abs(float(d1[3]) - float(d2[3])) * 0.2
    distance = age_dist + hypertension_dist + heart_disease_dist + residence_type_dist + avg_glucose_level_dist + bmi_dist + smoking_status_dist
    return distance