import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def get_data(url, columns, class_labels):
    df = pd.read_csv(url, names=columns, delim_whitespace=True)
    df['Class'] = df['Class'].map(class_labels)
    return df

def split_train_test(df, test_size=0.3, random_state=42):
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train['Class'] = y_train
    return X_train, X_test, y_train, y_test

def build_model(train_data, structure):
    model = BayesianNetwork(structure)
    model.fit(train_data, estimator=MaximumLikelihoodEstimator)
    return model

def predict_classes(bn_model, test_data):
    infer = VariableElimination(bn_model)
    preds = []
    for _, sample in test_data.iterrows():
        evidence = sample.to_dict()
        result = infer.map_query(variables=['Class'], evidence=evidence)
        preds.append(result['Class'])
    return preds

def calculate_accuracy(preds, true_labels):
    return accuracy_score(true_labels, preds)

def run_classifier():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data'
    columns = ['Class', 'T3', 'T4', 'TSH']
    class_labels = {1: 'Hyperthyroid', 2: 'Hypothyroid', 3: 'Normal'}

    data = get_data(url, columns, class_labels)
    X_train, X_test, _, y_test = split_train_test(data)
    network_structure = [('T3', 'Class'), ('T4', 'Class'), ('TSH', 'Class')]

    model = build_model(X_train, network_structure)
    predictions = predict_classes(model, X_test)
    
    acc = calculate_accuracy(predictions, y_test)
    print(f'Model accuracy: {acc * 100:.2f}%')

if __name__ == "__main__":
    run_classifier()
