import json
import joblib
import argparse
import numpy as np
import pandas as pd
from model import Model
from preprocessor import Preprocessor

class Pipeline:
    def __init__(self):
        self.processor = Preprocessor()
        self.model = Model()

    def run(self, data_path, test_mode=False, save_model=True):
        target = "In-hospital_death"
        data = pd.read_csv(data_path)
        if test_mode:
            # Load preprocessor and model if in testing mode
            if test_mode:
                try:
                    self.processor = joblib.load("preprocessor.pkl")
                    self.model = joblib.load("model.pkl")
                except FileNotFoundError:
                    print("Error: Preprocessor or Model files not found. Make sure to run in training mode first.")
                    return

            # Transform test data using preprocessor
            X_test = self.processor.transform(data, test_mode = True)
            
            # Predict probabilities and save results
            predict_probas = self.model.predict_proba(X_test)
            result = {'threshold': np.float64(self.model.threshold),
                       'predict_probas': predict_probas.tolist()}

            with open('Predictions.json', 'w') as result_file:
                json.dump(result, result_file)
        else:
            # Fit preprocessor and model in training mode
            self.processor = Preprocessor()
            self.processor.fit(data)
            X_train, y_train = self.processor.transform(data)
            self.model.fit(X_train, y_train)
            
            # Save preprocessor and model if specified
            if save_model:
                joblib.dump(self.processor, filename='preprocessor.pkl')
                joblib.dump(self.model, filename='model.pkl')


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run the data processing and modeling pipeline.")
    parser.add_argument("--data_path", type=str,
                        help="Path to the training or testing data.")
    parser.add_argument("--test", action="store_true",
                        help="Run the model in testing mode. If not specified, it runs in training mode.")
    args = parser.parse_args()

    if not args.data_path:
        print("Error: Please provide the path to the data using --data_path.")
    else:
        data_path = args.data_path
        test_mode = args.test
        pipeline = Pipeline()

        if test_mode:
            print("Test mode")
            pipeline.run(data_path=data_path, test_mode=True, save_model=False)
        else:
            print("Training mode")
            pipeline.run(data_path=data_path, test_mode=False, save_model=True)


