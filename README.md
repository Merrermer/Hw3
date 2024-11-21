# Hw3

# Brain Imaging Classification Pipeline

This repository provides a pipeline for classifying brain imaging data using Support Vector Machines (SVM) with Principal Component Analysis (PCA) for dimensionality reduction. The code is designed to process neuroimaging data, perform necessary preprocessing steps, select features, and evaluate the performance of the classifier using cross-validation and a test set.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Data Loading](#data-loading)
- [Preprocessing](#preprocessing)
- [Feature Selection](#feature-selection)
- [Model Selection](#model-selection)
- [Cross-Validation](#cross-validation)
- [Testing the Model](#testing-the-model)
- [Main Function](#main-function)
- [Usage](#usage)

## Overview

The pipeline includes the following key steps:

1. **Data Loading**: Load brain imaging data and corresponding labels.
2. **Preprocessing**: Apply a brain mask to extract relevant voxels.
3. **Feature Selection**: Use PCA to reduce dimensionality.
4. **Model Selection**: Define and train an SVM classifier.
5. **Cross-Validation**: Evaluate the model using k-fold cross-validation.
6. **Testing**: Assess the model's performance on a separate test set.

## Dependencies

Ensure you have the following Python packages installed:

- `numpy`
- `nibabel`
- `scipy`
- `scikit-learn`

You can install them using `pip`:

```bash
pip install numpy nibabel scipy scikit-learn
```

## Data Loading

The `get_data` function loads the brain imaging data and labels:

```python
def get_data(filepath, labelpath='label.mat'):
    img = nib.load(filepath)
    data = img.get_fdata()
    label = io.loadmat(labelpath)
    labels = label['label']
    return data, labels
```

- **Inputs**:
  - `filepath`: Path to the neuroimaging data file.
  - `labelpath`: Path to the label file (`.mat` format).

- **Outputs**:
  - `data`: The loaded imaging data.
  - `labels`: Corresponding labels for classification.

## Preprocessing

The `preprocessing` function applies a brain mask to filter out non-brain voxels:

```python
def preprocessing(data, threshold=1):
    brain_mask = np.mean(data, axis=-1) > threshold
    masked_data = data[brain_mask]
    return masked_data, brain_mask
```

- **Process**:
  - Calculates the mean intensity across all samples for each voxel.
  - Creates a mask where the mean intensity exceeds a threshold.
  - Applies the mask to retain only relevant voxels.

## Feature Selection

The `feature_selection` function reduces the dimensionality of the data using PCA:

```python
def feature_selection(data, n_components=50):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return pca, reduced_data
```

- **Process**:
  - Initializes PCA with the desired number of components.
  - Fits PCA on the data and transforms it to the reduced feature space.

## Model Selection

The `model_selection` function scales the data and defines the SVM classifier:

```python
def model_selection(reduced_data, kernel='linear', C=1):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(reduced_data)
    svm = SVC(kernel=kernel, C=C)
    return svm, scaler, data_scaled
```

- **Process**:
  - Scales the features to have zero mean and unit variance.
  - Initializes an SVM classifier with specified kernel and regularization parameter `C`.

## Cross-Validation

The `cross_val` function performs k-fold cross-validation:

```python
def cross_val(model, data_scaled, labels, k=4):
    kfold = KFold(n_splits=k, shuffle=True, random_state=123)
    scores = cross_val_score(model, data_scaled, labels, cv=kfold, scoring='accuracy')
    return scores
```

- **Process**:
  - Splits the data into `k` folds.
  - Evaluates the model's accuracy on each fold.
  - Returns the cross-validation scores.

## Testing the Model

The `test` function evaluates the model on the test set:

```python
def test(X_train_scaled, X_test_scaled, y_train, y_test, model):
    model.fit(X_train_scaled, np.ravel(y_train))
    test_accuracy = model.score(X_test_scaled, y_test)
    print('Accuracy on test set: {:.2f}%'.format(test_accuracy * 100))
```

- **Process**:
  - Trains the model on the entire training set.
  - Computes the accuracy on the test set.
  - Prints the test set accuracy.

## Main Function

The `brain_classification` function orchestrates the entire pipeline:

```python
def brain_classification(filepath, labelpath = 'label.mat', threshold = 1, n_components = 50, kernel = 'linear', C = 1.2, k = 4):

    # Load data
    data, labels = get_data(filepath = filepath)

    # Transpose data to have samples first
    data = data.transpose(3, 0, 1, 2)
    # data.shape: (184, 64, 64, 30)

    # Reshape data 
    data = data.reshape(184, -1)
    # data.shape: (184, 64*64*30)

    # Split data into training and test sets
    X_train_data, X_test_data, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    #Brain mask
    masked_data, brain_mask = preprocessing(X_train_data, threshold=threshold)  # Shape: (num_voxels, 184)
    masked_data_test = X_test_data[:, brain_mask]

    # Feature selection
    pca, reduced_data = feature_selection(masked_data, n_components=n_components)  # Shape: (184, 50)
    reduced_data_test = pca.transform(masked_data_test)

    # SVM model selection and data scaling
    svm, scaler, data_scaled = model_selection(reduced_data, kernel = kernel, C = C)
    data_scaled_test = scaler.transform(reduced_data_test)

    # Cross validation
    accuracy_scores = cross_val(svm, data_scaled, np.ravel(y_train), k=k)

    # Calculate final mean validation accuracy
    mean_accuracy = np.mean(accuracy_scores)

    print("Mean Validation Accuracy: {:.2f}%".format(mean_accuracy * 100))

    # Evaluate the model on test set
    test(data_scaled, data_scaled_test, y_train, y_test, svm)
```

- **Parameters**:
  - `filepath`: Path to the neuroimaging data file.
  - `labelpath`: Path to the label file.
  - `threshold`: Single threshold forr brain masking.
  - `n_components`: Number of principal components for PCA.
  - `kernel`: Kernel type for SVM (`'linear'`, `'rbf'`, etc.).
  - `C`: Regularization parameter for SVM.
  - `k`: Number of folds for cross-validation.

## Usage

To use the pipeline, call the `brain_classification` function with the appropriate file paths and parameters:

```python
brain_classification(filepath='path_to_brain_data.nii', labelpath='label.mat')
```

**Example**:

```python
if __name__ == "__main__":
    brain_classification(
        filepath='brain_data.nii',
        labelpath='label.mat',
        n_components=50,
        kernel='linear',
        C=1.2,
        k=4
    )
```

**Note**: Ensure that your data is correctly formatted and that the labels correspond to the data samples.

## Explanation of the Pipeline

1. **Data Preparation**:
   - The imaging data is loaded and reshaped so that each sample is a flattened vector of voxel intensities.
   - Data is split into training and test sets to evaluate the model's generalization ability.

2. **Preprocessing**:
   - A brain mask is applied to filter out irrelevant voxels based on intensity thresholding.
   - This reduces the dimensionality by keeping only significant voxels.

3. **Feature Extraction with PCA**:
   - PCA reduces the high-dimensional voxel data to a lower-dimensional feature space.
   - This step mitigates the curse of dimensionality and helps in capturing the most informative features.

4. **Model Training with SVM**:
   - An SVM classifier is trained on the reduced feature set.
   - The data is scaled to standardize feature values.

5. **Cross-Validation**:
   - K-fold cross-validation assesses the model's performance across different subsets of the training data.
   - The mean validation accuracy provides an estimate of how the model may perform on unseen data.

6. **Testing**:
   - The trained model is evaluated on the test set to obtain the final accuracy.
   - This step confirms the model's ability to generalize to new data.

## Conclusion

This pipeline provides a systematic approach to classifying brain imaging data using machine learning techniques. By combining preprocessing, dimensionality reduction, and robust evaluation methods, the model aims to achieve high accuracy in distinguishing between different classes in the dataset.

Feel free to adjust the parameters and experiment with different configurations to improve the model's performance.
