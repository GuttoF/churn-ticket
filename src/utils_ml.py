import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold


def highlight_max(s):
    """
    Just highlight the maximum in a Series
    """
    if s.dtype in [int, float]:
        is_max = s == s.max()
        return ["background-color: #F15854" if v else "" for v in is_max]
    else:
        return ["" for _ in s]


def metrics_models(
    models: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    verbose=True,
):
    """Return Metrics of the models

    Args:
        models (list): List of ML models for evaluation
        X_train (dataframe): Training variables
        y_train (list): Training target feature
        X_val (dataframe): Validation variables
        y_val (list): Validation target feature
        verbose (bool): Print training progress

    Returns:
        DataFrame: Dataframe with model evaluation scores
    """
    print("Please wait a moment - Doing ML")
    model_df = []

    for i, model in enumerate(models):
        model_name = type(model).__name__
        if verbose:
            print(f"Training model {i + 1}/{len(models)} -> {model_name}")
        model.fit(X_train, y_train)

        # Predict on the validation set
        yhat = model.predict(X_val)

        # Calculate metrics
        balanced_acc = balanced_accuracy_score(y_val, yhat)
        precision = precision_score(y_val, yhat)
        recall = recall_score(y_val, yhat)
        f1 = f1_score(y_val, yhat)
        roc_auc = roc_auc_score(y_val, yhat)

        df_result = pd.DataFrame(
            {
                "Model_Name": [model_name],
                "Balanced_Accuracy": [balanced_acc],
                "Precision": [precision],
                "Recall": [recall],
                "F1 Score": [f1],
                "ROCAUC": [roc_auc],
            }
        )

        model_df.append(df_result)

    final_result = pd.concat(model_df, ignore_index=True)
    print("Finished, check the results")

    return final_result


def metrics_cv(
    models: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold=0.5,
    verbose=True,
    kfold: int = 5,
):
    """Return Metrics of the models

    Args:
        models (list): List of ML models for evaluation
        X_train (dataframe): Training variables
        y_train (list): Training target feature
        threshold (float): Threshold for making predictions
        verbose (bool): Print training progress
        kfold (int): Number of folds for cross validation

    Returns:
        DataFrame: Dataframe with model evaluation scores
    """

    print("Please wait a moment - Doing CV")
    folds = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []

    model_df = []

    for i, model in enumerate(models):
        model_name = type(model).__name__
        if verbose:
            print(f"Folding model {i + 1}/{len(models)} -> {model_name}")

        for train_cv, val_cv in folds.split(X_train, y_train):
            X_train_fold = X_train.iloc[train_cv]
            y_train_fold = y_train.iloc[train_cv]
            X_val_fold = X_train.iloc[val_cv]
            y_val_fold = y_train.iloc[val_cv]

            model.fit(X_train_fold, y_train_fold)

            # Predict on the validation set with the specified threshold
            y_prob = model.predict_proba(X_val_fold)[:, 1]
            yhat = (y_prob >= threshold).astype(int)

            # Calculate metrics
            balanced_acc = balanced_accuracy_score(y_val_fold, yhat)
            acc_list.append(balanced_acc)

            precision = precision_score(y_val_fold, yhat)
            precision_list.append(precision)

            recall = recall_score(y_val_fold, yhat)
            recall_list.append(recall)

            f1 = f1_score(y_val_fold, yhat)
            f1_list.append(f1)

            roc_auc = roc_auc_score(y_val_fold, yhat)
            roc_auc_list.append(roc_auc)

        df_result = pd.DataFrame(
            {
                "Model_Name": [model_name],
                "Threshold": [threshold],
                "Balanced_Accuracy Mean": [np.mean(acc_list).round(3)],
                "Balanced_Accuracy STD": [np.std(acc_list).round(3)],
                "Precision Mean": [np.mean(precision_list).round(3)],
                "Precision STD": [np.std(precision_list).round(3)],
                "Recall Mean": [np.mean(recall_list).round(3)],
                "Recall STD": [np.std(recall_list).round(3)],
                "F1 Score Mean": [np.mean(f1_list).round(3)],
                "F1 Score STD": [np.std(f1_list).round(3)],
                "ROCAUC Mean": [np.mean(roc_auc_list).round(3)],
                "ROCAUC STD": [np.mean(roc_auc_list).round(3)],
            }
        )

        model_df.append(df_result)
        cv_result = pd.concat(model_df, ignore_index=True)

    print("Finished, check the results")

    return cv_result


def plot_confusion_matrix(models: list, X_val: pd.DataFrame, y_val: pd.Series):
    """Plot confusion matrices for multiple models.

    Args:
        models (list): List of ML models for evaluation.
        X_val (dataframe): Validation variables.
        y_val (list): Validation target feature.

    Returns:
        None
    """
    num_models = len(models)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))

    if num_models == 1:
        axes = [axes]  # Convert to a list if there's only one model

    for i, model in enumerate(models):
        model_name = type(model).__name__
        y_pred = model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Churn", "Churn"],
            yticklabels=["Not Churn", "Churn"],
            ax=axes[i],
        )

        axes[i].set_title(f"Confusion Matrix - {model_name}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    plt.show()


def plot_roc_auc(
    models: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    """Plot ROC AUC curves for multiple models.

    Args:
        models (list): List of ML models for evaluation.
        X_train (dataframe): Training variables.
        y_train (list): Training target feature.
        X_val (dataframe): Validation variables.
        y_val (list): Validation target feature.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))

    for model in models:
        model_name = type(model).__name__
        model.fit(X_train, y_train)

        # Predict the probabilities on the validation set
        y_prob = model.predict_proba(X_val)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC AUC Curves for Multiple Models")
    plt.legend(loc="lower right")
    plt.show()


def plot_best_threshold_roc(models: list, X_val: pd.DataFrame, y_val: pd.Series):
    """Plot ROC AUC curves with the best threshold for a list of models.

    Args:
        models (list): List of ML models.
        X_val (dataframe): Validation variables.
        y_val (list): Validation target feature.

    Returns:
        None
    """

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))

    if len(models) == 1:
        axes = [axes]  # Convert to a list if there's only one model

    for i, model in enumerate(models):
        model_name = type(model).__name__
        y_prob = model.predict_proba(X_val)[:, 1]

        # Find the best threshold using Youden's J statistic
        fpr, tpr, thresholds = roc_curve(y_val, y_prob)
        youden = tpr - fpr
        best_threshold = thresholds[np.argmax(youden)]

        y_pred = (y_prob >= best_threshold).astype(int)

        roc_auc = auc(fpr, tpr)

        label = f"Best Threshold = {best_threshold:.2f}\nAUC = {roc_auc:.2f}"
        axes[i].plot(fpr, tpr, lw=2, label=label)

        axes[i].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel("False Positive Rate")
        axes[i].set_ylabel("True Positive Rate")
        axes[i].set_title(f"ROC Curve - {model_name}")
        axes[i].legend(loc="lower right")

    plt.show()


def threshold_tuning_plot(
    models: list, X_val: pd.DataFrame, y_val: pd.Series, thresholds: list = [0.5]
):
    """Plot ROC AUC curves and confusion matrices for different thresholds in a list of models.

    Args:
        models (list): List of ML models.
        X_val (dataframe): Validation variables.
        y_val (list): Validation target feature.
        thresholds (list, optional): List of thresholds to evaluate (default: [0.5]).

    Returns:
        None
    """

    class_names = ["Not Churn", "Churn"]
    if thresholds is None:
        thresholds = [0.5]

    num_models = len(models)
    num_thresholds = len(thresholds)

    fig, axes = plt.subplots(
        num_models,
        num_thresholds + 1,
        figsize=(5 * (num_thresholds + 1), 5 * num_models),
    )
    plt.subplots_adjust(hspace=0.5)

    if num_thresholds == 1:
        axes = [axes]  # Convert to a list if there's only one threshold

    for j, model in enumerate(models):
        fpr_list, tpr_list, auc_list, cm_list = [], [], [], []

        for i, threshold in enumerate(thresholds):
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            fpr, tpr, _ = roc_curve(y_val, y_prob)
            roc_auc = auc(fpr, tpr)

            cm = confusion_matrix(y_val, y_pred)

            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(roc_auc)
            cm_list.append(cm)

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[j, i],
            )

            axes[j, i].set_title(
                f"{type(model).__name__}\nThreshold = {threshold:.2f}\nAUC = {roc_auc:.2f}"
            )
            axes[j, i].set_xlabel("Predicted")
            axes[j, i].set_ylabel("True")

        # Plot ROC AUC curve
        axes[j, -1].plot(fpr_list[-1], tpr_list[-1], color="darkorange", lw=2)
        axes[j, -1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axes[j, -1].set_xlim([0.0, 1.0])
        axes[j, -1].set_ylim([0.0, 1.05])
        axes[j, -1].set_xlabel("False Positive Rate")
        axes[j, -1].set_ylabel("True Positive Rate")
        axes[j, -1].set_title("ROC Curve")

    plt.show()

    return None


def threshold_tuning_plot_single(
    model, X_val: pd.DataFrame, y_val: pd.Series, threshold: float = 0.5
):
    """Plot ROC AUC curve and confusion matrix for a single model at a specified threshold.

    Args:
        model: The machine learning model.
        X_val (pd.DataFrame): Validation variables.
        y_val (pd.Series): Validation target feature.
        threshold (float, optional): Threshold to evaluate (default: 0.5).

    Returns:
        dict: A dictionary containing the confusion matrix and the ROC AUC curve data.
    """

    class_names = ["Not Churn", "Churn"]

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)

    # Compute confusion matrix
    cm = confusion_matrix(y_val, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix\nThreshold = {threshold:.2f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Plot ROC curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    # Return the confusion matrix and ROC curve data
    return {
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc
        }
    }
