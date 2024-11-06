from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics 
import joblib
from tqdm import tqdm 
from sklearn.svm import SVC

def read_files(training_path, testing_path):
    df_train = pd.read_csv(training_path, sep=",")
    df_test = pd.read_csv(testing_path, sep=",")
    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    return df_train,df_test

def load_data(training_path,test_path):
    df_train, df_test = read_files(training_path,test_path)

    # Select only the relevant columns
    svc = 0.5
    features = [
        f'Eigenvalues sum ({svc})', f'Omnivariance ({svc})', f'Eigenentropy ({svc})', 
        f'Anisotropy ({svc})', f'Planarity ({svc})', f'Linearity ({svc})', 
        f'PCA1 ({svc})', f'PCA2 ({svc})', f'Surface variation ({svc})', 
        f'Sphericity ({svc})', f'Verticality ({svc})', 
        f'1st eigenvalue ({svc})', f'2nd eigenvalue ({svc})', f'3rd eigenvalue ({svc})'
    ]

    label = 'Scalar field'

    # Separate the features (X_train) and the label (y_train) from the new training file
    X_train = df_train[features]
    y_train = df_train[label]

    # Separate the features (X_test) and the label (y_test) from the test file
    X_test = df_test[features]
    y_test = df_test[label]

    return X_train, y_train, X_test, y_test


def train_random_forest(X_train, y_train, n):
    clf = RandomForestClassifier(n_estimators = n, verbose=2) 
    clf.fit(X_train, y_train)

     # Save the model to a file
    model_filename = f'random_forest_model_{n}.pkl'
    joblib.dump(clf, model_filename)
    print(f"Model saved to {model_filename}")

    return clf 

def train_SVM(X_train, y_train):
    # Create an SVM classifier
    clf = SVC(kernel="rbf", gamma=0.5, C=1.0, verbose=2)
    clf.fit(X_train, y_train)

     # Save the model to a file
    model_filename = 'SVM.pkl'
    joblib.dump(clf, model_filename)
    print(f"Model saved to {model_filename}")

    return clf 


def get_metrics_RM(clf, X_train, y_train, X_test, y_test):

    # performing predictions on the test dataset
    y_pred = clf.predict(X_test)

    # using metrics module for accuracy calculation
    print("ACCURACY ON TESTING SET:", metrics.accuracy_score(y_test, y_pred),"\n")

    # Perform predictions on the training dataset
    y_train_pred = clf.predict(X_train)

    # Metrics to evaluate how well the model is performing on the training set
    accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
    print("ACCURACY ON TRAINING SET:", accuracy_train,"\n")

    # If you want to see the confusion matrix or other metrics:
    cm = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix on Testing Set:\n", cm,"\n")

    # Optionally, you could print out classification report
    classification_report_train = metrics.classification_report(y_test, y_pred)
    print("Classification Report on Testing Set:\n", classification_report_train)

    # Compute IoU for each class
    IoUs = []
    supports = []  # This will hold the support (number of true instances) for each class
    for i in range(len(cm)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        IoUs.append(IoU)
        supports.append(cm[i, :].sum())  # Support is the sum of true instances for class i
        # print(f"IoU for class {i}: {IoU:.4f}")

    # Calculate Macro Average IoU
    macro_avg_IoU = sum(IoUs) / len(IoUs)
    print(f"Macro Average IoU: {macro_avg_IoU:.4f}")

    # Calculate Weighted Average IoU
    weighted_avg_IoU = sum(IoU * support for IoU, support in zip(IoUs, supports)) / sum(supports)
    print(f"Weighted Average IoU: {weighted_avg_IoU:.4f}")

    
def create_pointCloud(clf, X_test, y_test,training_path,test_path, output_file_path="untermaederbrunnen_classified_points.txt"):
    df_train, df_test = read_files(training_path,test_path)
    # performing predictions on the test dataset
    y_pred = clf.predict(X_test)
    
    print("Performed Preditions")

    # Get the XYZ coordinates and original RGB values
    xyz = df_test[['//X', 'Y', 'Z']].values
    
    # Define color mapping (you can add more classes if needed)
    color_map = {
        1: (0, 0, 255),      # Blue - man-made terrain
        2: (0, 128, 255),    # Light Blue - natural terrain
        3: (0, 255, 0),      # Green - high vegetation
        4: (128, 255, 0),    # Yellow-Green - low vegetation
        5: (255, 255, 0),    # Yellow - buildings
        6: (255, 165, 0),    # Orange - hard scape
        7: (255, 128, 128),  # Light Red - scanning artifacts 
        8: (255, 0, 0)       # Red - cars 
    }
    
    # Initialize a list to hold the output data
    output_data = []
    
    for i, pred in enumerate(tqdm(y_pred, desc="Processing Predictions")):
        x, y, z = xyz[i]
        r, g, b = color_map.get(pred, (255, 255, 255))  # Default to white if label is out of bounds
        actual_label = y_test.iloc[i]  # Get the actual label
        output_data.append(f"{x},{y},{z},{r},{g},{b},{pred},{actual_label}\n")
    
    # Write the classified points to a txt file
    with open(output_file_path, "w") as file:
        file.writelines(output_data)
    
    print(f"Classified point cloud saved to {output_file_path}")



if __name__ == "__main__":
    training_path = 'Point Clouds/combined_training_data2.txt'
    testing_path = ['Point Clouds/bildstein_station5_xyz_intensity_rgb_filtered - Cloud.txt',
                    'Point Clouds/domfountain_station2_xyz_intensity_rgb_filtered - Cloud.txt',
                    'Point Clouds/neugasse_station1_xyz_intensity_rgb_filtered - Cloud.txt',
                    'Point Clouds/untermaederbrunnen_station3_xyz_intensity_rgb_filtered - Cloud.txt']

    X_train, y_train, X_test, y_test = load_data(training_path, testing_path[1])

    # clf = train_SVM(X_train, y_train)
    # clf = train_random_forest(X_train, y_train, 2)

    # example to load trained file
    # clf = joblib.load("SVM.pkl")
    clf = joblib.load("random_forest_model_2.pkl")

    print("Loaded Random Forest Model")

    get_metrics_RM(clf, X_train, y_train, X_test, y_test)

    # create_pointCloud(clf,X_test,y_test,training_path,testing_path[3])




