import pandas as pd
import seaborn as sns
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
def get_model_and_input():
    test, train = pd.read_csv("./test.csv"), pd.read_csv("./train.csv")
    train_id, test_id = train["PassengerId"], test["PassengerId"]
    train.drop("PassengerId",axis=1,inplace=True), test.drop("PassengerId",axis=1,inplace=True)
    target = train["Survived"]
    train.drop("Survived",axis=1,inplace=True)
    train.Age[train.Age.isnull()] = train.Age[train.Age.notnull()].median()
    test.Age[test.Age.isnull()] = test.Age[test.Age.notnull()].median()
    train.Embarked[train.Embarked.isnull()] = train.Embarked.mode()[0]
    test[test.Pclass==3].Fare.mean(), test[test.Pclass==2].Fare.mean(), test[test.Pclass==1].Fare.mean()
    test.Fare[test.Fare.isnull()] = test[test.Pclass==3].Fare.mean()
    categorical = ["Pclass","Sex","SibSp","Parch","Embarked"]
    numerical = ["Age","Fare"]
    train_size = train.shape[0]
    one_hot = pd.get_dummies(pd.concat([train[categorical],test[categorical]],ignore_index=True), columns=categorical)
    train_one_hot, test_one_hot = one_hot[:train_size], one_hot[train_size:]
    train_numerical, test_numerical = train[numerical], test[numerical]
    train_numerical.Fare, test_numerical.Fare = np.log1p(train_numerical.Fare), np.log1p(test_numerical.Fare)
    from sklearn.preprocessing import StandardScaler
    numerical_scaler = StandardScaler()
    train_numerical = pd.DataFrame(numerical_scaler.fit_transform(train_numerical),index = train_numerical.index, columns = train_numerical.columns)
    test_numerical = pd.DataFrame(numerical_scaler.transform(test_numerical),index = test_numerical.index, columns = test_numerical.columns)
    test_numerical.index = test_one_hot.index
    train_final, test_final = \
        pd.concat([train_one_hot, train_numerical],axis=1), pd.concat([test_one_hot, test_numerical],axis=1)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(train_final.drop("Fare",axis=1), target)

    return model, train_final.drop("Fare",axis=1), test_final.drop("Fare",axis=1)