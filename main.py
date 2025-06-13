#!/usr/bin/env python3
"""
AsymmeTree CLI Interface

A command-line interface to demonstrate and use AsymmeTree for imbalanced classification tasks.
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from asymmetree import AsymmeTree


def create_demo_data():
    """Create a sample imbalanced dataset for demonstration."""
    print("📊 Creating demo imbalanced dataset...")

    np.random.seed(42)
    n_samples = 1000

    # Create features
    age = np.random.normal(35, 10, n_samples)
    income = np.random.exponential(50000, n_samples)
    credit_score = np.random.normal(650, 100, n_samples)

    # Create imbalanced target (5% positive class)
    # Higher age, lower income, and lower credit score increase fraud probability
    fraud_prob = (
        (age > 40) * 0.1
        + (income < 30000) * 0.15
        + (credit_score < 600) * 0.2
        + np.random.random(n_samples) * 0.05
    )

    y = (fraud_prob > 0.45).astype(int)

    # Create DataFrame
    X = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "account_type": np.random.choice(
                ["premium", "standard", "basic"], n_samples
            ),
            "region": np.random.choice(["north", "south", "east", "west"], n_samples),
        }
    )

    print(
        f"✅ Dataset created: {len(X)} samples, {y.sum()} positive cases ({y.mean():.1%} positive rate)"
    )
    return X, y


def run_demo_mode():
    """Run a demonstration of AsymmeTree capabilities."""
    print("🚀 Running AsymmeTree Demo Mode")
    print("=" * 50)

    # Create demo data
    X, y = create_demo_data()

    # Demo 1: Basic automatic tree
    print("\n1️⃣ Basic Automatic Tree Building")
    print("-" * 30)

    tree1 = AsymmeTree(
        max_depth=3,
        sorted_by="f_score",
        node_min_recall=0.05,
        leaf_min_precision=0.1,
        verbose=True,
    )

    tree1.fit(X, y, cat_features=["account_type", "region"], auto=True)
    print("\n📈 Performance Metrics:")
    tree1.performance()

    # # Demo 2: Interactive mode example
    # print("\n2️⃣ Interactive Tree Building Example")
    # print("-" * 30)

    # tree2 = AsymmeTree(max_depth=3)
    # tree2.import_data(X, y, cat_features=["account_type", "region"])
    # tree2.fit(auto=False)

    # print("🌳 Tree structure after fitting:")
    # tree2.print(show_metrics=True)

    # Demo 3: Export capabilities
    print("\n2️⃣ Export Capabilities")
    print("-" * 30)

    print("📄 SQL Export:")
    sql_rules = tree1.to_sql()
    print(sql_rules)

    print("\n📊 Tree Dictionary Export:")
    tree_dict = tree1.to_dict()
    print(f"Tree has {len(tree_dict)} nodes")

    return tree1


def run_interactive_mode():
    """Run interactive mode for building trees step by step."""
    print("🔧 Interactive AsymmeTree Mode")
    print("=" * 50)

    # Get data source
    data_choice = input(
        "Choose data source:\n1. Demo data\n2. Load from CSV\nEnter choice (1-2): "
    )

    if data_choice == "2":
        csv_path = input("Enter path to CSV file: ")
        target_col = input("Enter target column name: ")
        try:
            data = pd.read_csv(csv_path)
            y = data[target_col]
            X = data.drop(columns=[target_col])
            print(f"✅ Loaded {len(X)} samples from {csv_path}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return
    else:
        X, y = create_demo_data()

    # Configure tree parameters
    print("\n🔧 Tree Configuration")
    max_depth = int(input("Max depth (default 5): ") or "5")
    sorted_by = (
        input("Sorting criterion (f_score/ig/igr/iv) [default: f_score]: ") or "f_score"
    )

    # Get categorical features
    cat_features = []
    print(f"\nAvailable features: {list(X.columns)}")
    cat_input = input(
        "Categorical features (comma-separated, or press Enter for auto-detect): "
    )
    if cat_input:
        cat_features = [f.strip() for f in cat_input.split(",")]
    else:
        # Auto-detect categorical features
        cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Using categorical features: {cat_features}")

    # Initialize tree
    tree = AsymmeTree(
        max_depth=max_depth,
        sorted_by=sorted_by,
        verbose=True,
    )

    tree.import_data(X, y, cat_features=cat_features)

    # Interactive loop
    while True:
        print("\n🌳 Current Tree Structure:")
        tree.print(show_metrics=True)

        print("\nOptions:")
        print("1. Split a node")
        print("2. Quick split with custom condition")
        print("3. Continue auto-fit from node")
        print("4. Show performance")
        print("5. Export to SQL")
        print("6. Save and exit")

        choice = input("Enter choice (1-6): ")

        if choice == "1":
            node_id = int(input("Enter node ID to split: "))
            tree.split(id=node_id)

        elif choice == "2":
            node_id = int(input("Enter node ID: "))
            condition = input("Enter SQL condition (e.g., 'age >= 25'): ")
            overwrite = input("Overwrite existing split? (y/n): ").lower() == "y"
            tree.quick_split(id=node_id, sql=condition, overwrite=overwrite)

        elif choice == "3":
            node_id = int(input("Enter node ID to continue from: "))
            tree.continue_fit(id=node_id)

        elif choice == "4":
            tree.performance()

        elif choice == "5":
            sql = tree.to_sql()
            print(f"SQL Rules:\n{sql}")

        elif choice == "6":
            save_path = input("Enter save path (or press Enter to skip): ")
            if save_path:
                tree.save(save_path)
                print(f"✅ Tree saved to {save_path}")
            break

        else:
            print("Invalid choice. Please try again.")


def run_custom_mode(args):
    """Run with custom dataset and parameters."""
    print("📁 Custom Dataset Mode")
    print("=" * 50)

    # Load data
    try:
        data = pd.read_csv(args.data)
        y = data[args.target]
        X = data.drop(columns=[args.target])
        print(f"✅ Loaded {len(X)} samples from {args.data}")
        print(f"Positive rate: {y.mean():.1%}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Configure tree
    tree_params = {
        "max_depth": args.max_depth,
        "sorted_by": args.criterion,
        "verbose": True,
    }

    if args.min_precision:
        tree_params["leaf_min_precision"] = args.min_precision

    if args.min_recall:
        tree_params["node_min_recall"] = args.min_recall

    categorical_features = (
        args.categorical_features.split(",") if args.categorical_features else None
    )

    # Build tree
    tree = AsymmeTree(**tree_params)
    tree.fit(X, y, cat_features=categorical_features, auto=True)

    # Show results
    print("\n📈 Performance:")
    tree.performance()

    print("\n🌳 Tree Structure:")
    tree.print(show_metrics=True)

    # Export if requested
    if args.output_sql:
        sql = tree.to_sql()
        with open(args.output_sql, "w") as f:
            f.write(sql)
        print(f"✅ SQL rules saved to {args.output_sql}")

    if args.output_model:
        tree.save(args.output_model)
        print(f"✅ Model saved to {args.output_model}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="AsymmeTree: Interactive Asymmetric Decision Trees for Imbalanced Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo                    # Run demonstration mode
  python main.py interactive             # Interactive tree building
  python main.py custom --data data.csv --target fraud --max-depth 5
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operating modes")

    # Demo mode
    demo_parser = subparsers.add_parser(
        "demo", help="Run demonstration with sample data"
    )

    # Interactive mode
    interactive_parser = subparsers.add_parser(
        "interactive", help="Interactive tree building"
    )

    # Custom mode
    custom_parser = subparsers.add_parser("custom", help="Use custom dataset")
    custom_parser.add_argument("--data", required=True, help="Path to CSV file")
    custom_parser.add_argument("--target", required=True, help="Target column name")
    custom_parser.add_argument(
        "--max-depth", type=int, default=5, help="Maximum tree depth"
    )
    custom_parser.add_argument(
        "--criterion",
        choices=["f_score", "ig", "igr", "iv"],
        default="f_score",
        help="Splitting criterion",
    )
    custom_parser.add_argument(
        "--categorical-features", help="Comma-separated categorical features"
    )
    custom_parser.add_argument(
        "--min-precision", type=float, help="Minimum leaf precision"
    )
    custom_parser.add_argument("--min-recall", type=float, help="Minimum node recall")
    custom_parser.add_argument("--output-sql", help="Output SQL rules to file")
    custom_parser.add_argument("--output-model", help="Save model to file")

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo_mode()
    elif args.mode == "interactive":
        run_interactive_mode()
    elif args.mode == "custom":
        run_custom_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
