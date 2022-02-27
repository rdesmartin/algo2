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

# Kmeans ++

Nous avons implémenté Kmeans++ en utilisant la méthode `choice` de numpy: 
celle-ci permet de choisir parmi une liste de valeurs avec une loi de probabilité donnée.
Ici, chaque point a une probabilité d'être choisi comme centroïde proportionnelle à sa 
distance avec le centroïde le plus proche; plus le centroïde est proche, plus la 
probabilité que le point soit choisi est faible et inversement. 


# Comparaison entre la méthode du coude et la méthode de Silhouette 
## à mettre après la description de chaque méthode)
L'objectif était de comparer deux heuristiques et de vérifier si elles nous donnaient le même 
nombre optimal de clusters pour la méthode de clustering k-means. Cette expérience était
intéressante car le dataset étudié est constitué de données réelles et il n'y a pas de 
clusters artificiels dans la distribution des points.
Si les deux heuristiques nous donnent un même nombre de clusters optimal, cela conforterait le choix du nombre de clusters optimal. 
Dans le cas inverse, il faudrait nuancer le résultat.

La méthode du coude nous donne un nombre optimal de cluster de 3. 
Malheureusement, la méthode silhouette n'a pas donné les résultats espérés. Le coefficient de silhouette est toujours aux alentours de 0.6 quel que soit
le nombre de clusters essayé ; il y a là spurement une erreur d'implémentation qui nous empêche de comparer les résultats des deux méthodes.
