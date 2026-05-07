import joblib
from pathlib import Path

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from modeling_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():
    # Paths
    db_path = r"D:\data science\inventory invoice intellignece ml\data\inventory.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = load_vendor_invoice_data(db_path)

    # Prepare data
    X, Y = prepare_features(df)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # Train models
    lr_model = train_linear_regression(X_train, Y_train)
    dt_model = train_decision_tree(X_train, Y_train)
    rf_model = train_random_forest(X_train, Y_train)

    # Evaluate models
    results = []

    results.append(
        evaluate_model(lr_model, X_test, Y_test, "Linear Regression")
    )
    results.append(
        evaluate_model(dt_model, X_test, Y_test, "Decision Tree Regression")
    )
    results.append(
        evaluate_model(rf_model, X_test, Y_test, "Random Forest Regression")
    )

    # ✅ FIXED BUG HERE
    best_model_info = min(results, key=lambda x: x["MAE"])
    best_model_name = best_model_info["model"]

    # Map models
    model_mapping = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }

    best_model = model_mapping[best_model_name]

    # Save model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()