import sys
import sys
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'plot.png')

def process_xes_file(file_path):
    log = xes_importer.apply(file_path)

    # Convert log to dataframe
    data = pd.DataFrame([{
        'case_id': trace.attributes['concept:name'],
        'event': event['concept:name'],
        'transition': event['lifecycle:transition'],
        'time_stamp': event['time:timestamp'],
        'life_cycle': event['lifecycle:transition']
    } for trace in log for event in trace])
    data = data.sort_values(by=['case_id', 'time_stamp'])

    data['time_since_last_event'] = data.groupby('case_id')['time_stamp'].diff().fillna(pd.Timedelta(seconds=0))
    data['time_since_last_event'] = data['time_since_last_event'].dt.total_seconds()

    # Calculate the total time of each case
    data['case_end_time'] = data.groupby('case_id')['time_stamp'].transform('max')
    data['time_to_end'] = (data['case_end_time'] - data['time_stamp']).dt.total_seconds()

    # Feature Engineering
    data['prev_event'] = data.groupby('case_id')['event'].shift(-1)
    data.dropna(subset=['prev_event'], inplace=True)

    # Encode categorical variables with separate encoders
    event_label_encoder = LabelEncoder()
    life_cycle_label_encoder = LabelEncoder()

    data['event_encoded'] = event_label_encoder.fit_transform(data['event'])
    data['prev_event_encoded'] = event_label_encoder.transform(data['prev_event'])
    data['life_cycle_encoded'] = life_cycle_label_encoder.fit_transform(data['life_cycle'])

    # Prepare train and test sets
    features = ['event_encoded', 'life_cycle_encoded', 'time_since_last_event']
    X = data[['time_since_last_event', 'prev_event_encoded']]
    y = data['event_encoded']

    X_log = pd.DataFrame(np.log1p(X))
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_transformed = scaler.fit_transform(X_log)

    class_weights = {0: 4207/5173, 1: 81/5173, 2: 875/5173, 3: 10/5173}
    X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.2,  random_state=42)

    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(class_weight=class_weights)
    rf_classifier.fit(X_train, y_train)
    rf_predictions = rf_classifier.predict(X_test)
    print("\n=== Random Forest Classifier Report ===")
    print(classification_report(y_test, rf_predictions, target_names=event_label_encoder.classes_, zero_division=0))
    plot_confusion_matrix(y_test, rf_predictions, event_label_encoder.classes_, 'Random_Forest')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python other_models_processor.py <path_to_xes_file>")
        sys.exit(1)

    xes_file_path = sys.argv[1]
    process_xes_file(xes_file_path)
