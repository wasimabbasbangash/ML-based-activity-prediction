import sys
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np

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

    class_weights = {0: 4207/5173, 1: 81/5173, 2: 875/5173, 3: 10/5173}  # Adjust the weights based on your class distribution

    X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.2, random_state=42)

    # Convert 'prev_event' to string and prepare for N-gram
    X_ngram = data['prev_event'].astype(str)
    y_ngram = data['event_encoded']
    X_train_ngram, X_test_ngram, y_train_ngram, y_test_ngram = train_test_split(X_ngram, y_ngram, test_size=0.2, random_state=42)

    ngram_model = make_pipeline(CountVectorizer(analyzer='word', ngram_range=(1, 2)), MultinomialNB())
    ngram_model.fit(X_train_ngram, y_train_ngram)
    ngram_predictions = ngram_model.predict(X_test_ngram)

    print("=== N-gram Model Classification Report ===")
    print(classification_report(y_test_ngram, ngram_predictions, target_names=event_label_encoder.classes_ ,zero_division=0))

    # Displaying the original and predicted activities as strings
    for i in range(len(y_test_ngram)):
        original_activity = event_label_encoder.inverse_transform([y_test_ngram.iloc[i]])[0]
        predicted_activity = event_label_encoder.inverse_transform([ngram_predictions[i]])[0]
        # print(f"Original: {original_activity}, Predicted: {predicted_activity}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_xes.py <path_to_xes_file>")
        sys.exit(1)

    xes_file_path = sys.argv[1]
    process_xes_file(xes_file_path)
