import streamlit as st
import numpy as np
import pickle
from models.logistic_regression import LogisticRegression
from libs.sense_proc import TextPreprocessor

# Define the custom DecisionTree class that was used to save the model
# This must exactly match the class that was used when saving the model
class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifierCustom:
    def __init__(self, max_depth=15, min_samples_split=10, max_thresholds=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split:
            return TreeNode(value=self._most_common_label(y))

        best_feat, best_thresh = self._best_split(X, y)

        if best_feat is None:
            return TreeNode(value=self._most_common_label(y))

        left_indices = X[:, best_feat] <= best_thresh
        right_indices = X[:, best_feat] > best_thresh
        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return TreeNode(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y):
        best_gain = -1
        best_index, best_thresh = None, None

        for feature_index in range(X.shape[1]):
            X_column = X[:, feature_index]
            unique_values = np.unique(X_column)

            if len(unique_values) > self.max_thresholds:
                thresholds = np.linspace(min(unique_values), max(unique_values), self.max_thresholds)
            else:
                thresholds = unique_values

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_index = feature_index
                    best_thresh = threshold

        return best_index, best_thresh

    def _information_gain(self, y, feature_column, threshold):
        parent_entropy = self._entropy(y)
        left_mask = feature_column <= threshold
        right_mask = feature_column > threshold

        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y[left_mask]), len(y[right_mask])
        e_l = self._entropy(y[left_mask])
        e_r = self._entropy(y[right_mask])
        weighted_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - weighted_entropy

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _most_common_label(self, y):
        from collections import Counter
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# Load vocab and idf
vocab = np.load("models_pretrained/vocab.npy", allow_pickle=True).tolist()
idf = np.load("models_pretrained/idf.npy", allow_pickle=True)

# Define available models
available_models = {
    "Logistic Regression": "models_pretrained/logistic_regression.pkl",
    "SVM": "models_pretrained/svm_model.pkl",
    "DecisionTree": "models_pretrained/dt.pkl"  # Updated path
}

# Define path to the vectorizer
dt_vectorizer_path = "models_pretrained/custom_tfidf_vectorizer.pkl"  # Updated path

# Setup preprocessor
preprocessor = TextPreprocessor()
preprocessor.vocab = vocab
preprocessor.idf_vector = idf

# Streamlit UI config
st.set_page_config(page_title="SenseNet2D: sentiment analyzer", page_icon="📝", layout="centered")
st.title("SenseNet2D: sentiment analyzer")

# Add a debug mode toggle
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Model selection dropdown
selected_model_name = st.selectbox(
    "Choose model:", 
    options=list(available_models.keys())
)

# User input
input_text = st.text_area("Enter your text here...", height=150)

# Test cases for quick testing
if st.sidebar.checkbox("Show Test Cases"):
    test_cases = [
        "I absolutely loved the product, it was amazing!",
        "I am sad and disappointed.",
        "Not sure what I feel about this.",
        "Worst thing ever!",
        "Such a beautiful day!",
    ]
    selected_test = st.sidebar.selectbox("Select a test case:", test_cases)
    if st.sidebar.button("Use Selected Test Case"):
        input_text = selected_test

# Handle form submission
if st.button("Classify"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Load the selected model
        model_path = available_models[selected_model_name]
        try:
            # Load the model based on its type
            if selected_model_name == "Logistic Regression":
                with open(model_path, "rb") as f:
                    model_params = pickle.load(f)
                model = LogisticRegression()
                model.W = model_params['weights']
                model.B = model_params['bias']
                
                # Preprocess using your custom preprocessor
                processed_text = preprocessor.preprocess(input_text)
                feature_vector = preprocessor.tfidf([processed_text])
                
                # Debug information
                if debug_mode:
                    st.write("Model type: Logistic Regression")
                    st.write(f"Feature vector shape: {feature_vector.shape}")
                
                # Predict
                prediction = model.predict(feature_vector)[0]
                sentiment = "Positive" if prediction == 1 else "Negative"
                
            elif selected_model_name == "SVM":
                # Load the SVM model components
                with open(model_path, "rb") as f:
                    svm_data = pickle.load(f)
                
                # Extract components
                svm_model = svm_data['model']
                vectorizer = svm_data['vectorizer']
                scaler = svm_data['scaler']
                
                # Use SVM's own preprocessing pipeline
                feature_vector_raw = vectorizer.transform([input_text]).toarray()
                feature_vector = scaler.transform(feature_vector_raw)
                
                # Debug information
                if debug_mode:
                    st.write("Model type: SVM")
                    st.write(f"Feature vector shape: {feature_vector.shape}")
                
                # Predict
                prediction = svm_model.predict(feature_vector)[0]
                
                # Convert back using label_encoder if available
                if 'label_encoder' in svm_data:
                    label_encoder = svm_data['label_encoder']
                    sentiment = label_encoder.inverse_transform([prediction])[0]
                else:
                    sentiment = "Positive" if prediction == 1 else "Negative"
                
            elif selected_model_name == "DecisionTree":
                # Load the Decision Tree model
                with open(model_path, "rb") as f:
                    dt_model = pickle.load(f)
                
                # Load the TF-IDF vectorizer specifically for Decision Tree
                try:
                    with open(dt_vectorizer_path, "rb") as f_vec:
                        dt_vectorizer = pickle.load(f_vec)
                    
                    if debug_mode:
                        st.write("Model type: Decision Tree")
                        st.write("Using paired TF-IDF vectorizer")
                    
                    # Transform input text using the specific vectorizer
                    feature_vector = dt_vectorizer.transform([input_text]).toarray()
                    
                    if debug_mode:
                        st.write(f"Feature vector shape: {feature_vector.shape}")
                        # Show sample of feature vector
                        st.write("Sample of feature vector (first 10 values):")
                        st.write(feature_vector[0][:10])
                        
                        # Check for zeros
                        non_zero = np.count_nonzero(feature_vector)
                        st.write(f"Non-zero features: {non_zero} out of {feature_vector.size}")
                    
                    # Make prediction
                    prediction = dt_model.predict(feature_vector)[0]
                    
                    if debug_mode:
                        st.write(f"Raw prediction value: {prediction}")
                    
                    # Map prediction to sentiment
                    sentiment = "Positive" if prediction == 1 else "Negative"
                    
                except FileNotFoundError:
                    st.error(f"Decision Tree vectorizer not found at {dt_vectorizer_path}")
                    st.info("Please make sure the vectorizer is saved in the models_pretrained directory")
                    raise
            
            # Display result with the model name
            st.header(f"Prediction: {sentiment}")
            st.info(f"Model used: {selected_model_name}")
            
        except FileNotFoundError:
            st.error(f"Model file not found: {model_path}")
            st.info("Make sure the model file is in the correct location")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
