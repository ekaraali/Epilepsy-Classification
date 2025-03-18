import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class EEGPreprocessor:
    def __init__(self, csv_path, z_score_threshold=2.5, feature_percent_threshold=0.7,
                 test_size=0.2, random_state=42):
        """
        Initializes the preprocessor with configuration parameters.
        """
        self.csv_path = csv_path
        self.z_score_threshold = z_score_threshold
        self.feature_percent_threshold = feature_percent_threshold
        self.test_size = test_size
        self.random_state = random_state

    def load_and_preprocess(self):
        """
        Loads the dataset, performs a stratified train-test split, and applies on the training set:
          1. Outlier elimination on raw data (using per-class statistics),
          2. Normalization on the filtered data,
          3. SMOTE oversampling.

        For the test set, only normalization (per class) is applied.

        Returns:
            X_train, y_train, X_test, y_test
        """
        # Load dataset and drop unwanted column
        eeg_df = pd.read_csv(self.csv_path)
        eeg_df = eeg_df.drop(["Unnamed: 0"], axis=1)

        # Stratified train-test split based on label 'y'
        train_df, test_df = train_test_split(
            eeg_df, test_size=self.test_size, random_state=self.random_state, stratify=eeg_df['y']
        )

        # Process training set for each class (outlier elimination then normalization)
        def process_class(df, class_label):
            # Extract raw features
            features = df.drop(columns=['y'])
            # Compute z-scores on raw data
            mean_vals = features.mean()
            std_vals = features.std()
            z_scores = (features - mean_vals) / std_vals

            # Identify rows with acceptable number of extreme values
            exceeds_threshold = np.abs(z_scores) > self.z_score_threshold
            num_exceeding_per_row = np.sum(exceeds_threshold, axis=1)
            total_features = features.shape[1]
            rows_to_keep = (num_exceeding_per_row / total_features) < self.feature_percent_threshold

            # Filter out the outlier rows
            filtered_features = features[rows_to_keep]

            # Now apply normalization on the cleaned data (recompute mean and std)
            norm_filtered = (filtered_features - filtered_features.mean()) / filtered_features.std()
            filtered_labels = np.full(norm_filtered.shape[0], class_label)
            return norm_filtered, filtered_labels

        # Process training data for class 3 and class 4
        train_class3 = train_df[train_df['y'] == 3].copy()
        train_class4 = train_df[train_df['y'] == 4].copy()

        norm_train_class3, labels_class3 = process_class(train_class3, 0)  # 0 represents class 3
        norm_train_class4, labels_class4 = process_class(train_class4, 1)  # 1 represents class 4

        print("Training class 3 rows after outlier elimination and normalization:", norm_train_class3.shape[0])
        print("Training class 4 rows after outlier elimination and normalization:", norm_train_class4.shape[0])

        # Combine training data and apply SMOTE
        X_train = pd.concat([norm_train_class3, norm_train_class4], ignore_index=True)
        y_train = np.concatenate([labels_class3, labels_class4])
        smote = SMOTE(random_state=self.random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print("Training set after SMOTE shape:", X_train_res.shape)

        # Process test set: only normalization per class
        def normalize_class(df):
            features = df.drop(columns=['y'])
            return (features - features.mean()) / features.std()

        test_class3 = test_df[test_df['y'] == 3].copy()
        test_class4 = test_df[test_df['y'] == 4].copy()

        norm_test_class3 = normalize_class(test_class3)
        norm_test_class4 = normalize_class(test_class4)

        X_test = pd.concat([norm_test_class3, norm_test_class4], ignore_index=True)
        y_test = np.concatenate([
            np.full(norm_test_class3.shape[0], 0),
            np.full(norm_test_class4.shape[0], 1)
        ])

        return X_train_res, y_train_res, X_test, y_test
