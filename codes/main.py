import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Assuming 'cardio' is the target variable
    X = df.drop(columns=['id', 'target'])
    y = df['target']
    
    # Keeping the index for later use
    indices = df.index.tolist()
    
    return X, y, indices


class TrainerModel:
    def __init__(self, X, y, threshold, model_number):
        self.X = X
        self.y = y
        self.threshold = threshold
        self.model_number = model_number  # Keep track of the model number
        self.model = None
        self.trained_indices = []  # To store indices of correctly trained samples
        self.X_baad = pd.DataFrame()  # DataFrame for incorrectly predicted samples
        self.y_baad = pd.Series()  # Series for incorrectly predicted labels

    def train(self):
        while True:  # Continue until the model meets the accuracy threshold
            # Create a new instance of LogisticRegression for each training iteration
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(self.X, self.y)

            # Calculate accuracy on the training data
            y_pred = self.model.predict(self.X)
            accuracy = accuracy_score(self.y, y_pred)
            print(f"Model {self.model_number} Accuracy: {accuracy:.4f} on {len(self.X)} data")
            
            # Update incorrectly predicted samples for next training iteration
            self.X_baad = self.X[y_pred != self.y]
            self.y_baad = self.y[y_pred != self.y]

            # Store indices for samples that this model trained on correctly
            self.trained_indices = self.X.index[y_pred == self.y].tolist()

            # Exclude incorrectly predicted samples from the training data
            self.X = self.X[y_pred == self.y]
            self.y = self.y[y_pred == self.y]

            # Check if the accuracy meets the threshold
            if accuracy >= self.threshold:
                print(f"Model {self.model_number} reached threshold of {self.threshold}.")
                break  # Exit the loop if accuracy meets the threshold

        return accuracy, self.X_baad, self.y_baad  # Return accuracy and bad data


class Trainer:
    def __init__(self, X, y, threshold, cutoff):
        self.models = []
        self.X = X
        self.y = y
        self.threshold = threshold
        self.cutoff = cutoff
        self.training_tracker = {}  # Dictionary to store which model trained on each sample

    def train(self, num=10):
        for i in range(num):
            print(f"Training model {i + 1}/{num}")
            model = TrainerModel(self.X, self.y, self.threshold, i + 1)
            
            # Train the model
            try:
                accuracy = model.train()
            except Exception as e:
                print(f"Error during training: {e}")
                break

            # Track which model trained each index
            for idx in model.trained_indices:
                if idx in self.training_tracker:
                    self.training_tracker[idx].append(i + 1)  # Append model number
                else:
                    self.training_tracker[idx] = [i + 1]

            # Update the training data
            self.X = model.X_baad
            self.y = model.y_baad
            self.models.append(model.model)

            print(f"Model {i + 1} trained. Remaining data: {len(self.X)} samples.")

            # Check if we reached the cutoff
            if len(self.X) < self.cutoff:
                print("Cutoff reached. Stopping training.")
                break

        print(f"Training completed. Total models trained: {len(self.models)}.")

class Predictor:
    def __init__(self, models, X_test, y_test, test_indices, training_tracker=None):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.test_indices = test_indices
        self.training_tracker = training_tracker  # Use training tracker from Trainer
        self.results_df = None
         

    def predict(self):
        results = []  # Temporary list to build the DataFrame
        y_pred_final = []  # List to store final predictions for accuracy metrics

        for index, x, y in zip(self.test_indices, self.X_test.to_numpy(), self.y_test):  # Using .to_numpy() to ensure we get the array
            row = {"Index": index}  # Initialize row with index and expected output
            probas = [model.predict_proba(x.reshape(1, -1))[0] for model in self.models]  # Probabilities for each model
            
            # Collect probabilities with model-specific naming
            for i, p in enumerate(probas):
                row[f"model{i}"] = (p[0], p[1])  # Tuple of probabilities (prob_0, prob_1) for each model
            
            # Calculate confidence and final prediction
            confidence = [abs(p[1] - 0.5) for p in probas]
            max_confidence_index = np.argmax(confidence)
            verdict = 1 if probas[max_confidence_index][1] > 0.5 else 0
            row["Expected"] = self.y_test[index]
            row["Prediction"] = verdict
            row["matches?"] = "YES" if verdict == y else "NO"

            y_pred_final.append(row["Prediction"])  # Append final prediction for metrics calculation
            
            results.append(row)

        self.results_df = pd.DataFrame(results)

        # Calculate and display testing metrics
        accuracy = accuracy_score(self.y_test, y_pred_final)
        report = classification_report(self.y_test, y_pred_final, output_dict=True)
        print(f"\nTesting Accuracy: {accuracy}")
        # print("\nDetailed Classification Report:\n", pd.DataFrame(report).transpose())

        # Save results to CSV excluding "Trained_in_Model" column
        self.results_df.drop(columns=["Trained_in_Model"], errors="ignore", inplace=True)
        self.results_df.to_excel("predictions_results.xlsx", index=False)
        print("Results saved to predictions_results.xlsx")
        

# Main execution
if __name__ == "__main__":
    file_path = '.\\data\\cardio.csv'
    
    X, y, indices = load_and_preprocess_data(file_path)  # Replace with your CSV file path
    
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )

    threshold = 0.95 # trains model until it reaches atleast 95% accuracy by excluding data
    cutoff = len(X_train) * 0.01 * 0.1 # no more training will take place is remaining dataset size reduces to 0.1% main dataset size
    
    # Training phase
    trainer = Trainer(X_train, y_train, threshold, cutoff)  # Adjust threshold and cutoff as needed
    trainer.train()
    
    # Prediction phase
    predictor = Predictor(trainer.models, X_test, y_test, test_indices)
    predictor.predict()
