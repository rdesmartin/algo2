# Normalisation of the dataset.

First, we splitted the features into 4 categories: 
- The critical features: Hypertension + heart diseases + age
- The average features: Avg clucose level + bmi + smoking status 
- The minor feature: Residence type
- The irrelevant features: gender, id, work type, ever married
We have discarded the 4th group and normalised each features.
Categorical features has been set with this dict:
`cleanup_nums = {"Residence_type":     {"Rural": 1, "Urban": 0},
                "smoking_status":     {"never smoked": 0, "Unknown": 0, "smokes": 1.5, "formerly smoked": 1.25 }}`
The proportion is set by a variable by category of features:
Critical features : 0.8
Average features: 0.5
Minor features : 0.2
Irrelevant features: 0.0

