import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import os
import json

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    #convert list -> np.array()
    inputs = np.array(data["features"])
    targets = np.array(data["mms"])

    print(inputs.shape, targets.shape)

    return inputs, targets

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2,p,dof,ex = ss.chi2_contingency(confusion_matrix)

    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2/min((k-1),(r-1)))

if __name__ == "__main__":
    data_path = os.path.abspath("json/data.json")
    inputs, targets = load_data(data_path=data_path)

    for val in range(0, len(inputs[0])):
        X = (inputs[1], inputs[2])
        Y = (targets[1], targets[2])
        model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, Y)
        rfe = RFE(model, 5)
        fit = rfe.fit(X, Y)

        print( fit.n_features_)
        print(f'Observing frame # {val}')
        print("Selected Features: %s"% fit.support_)
        print("Feature Ranking: %s"% fit.ranking_)

    #cramers_v(inputs, targets)
