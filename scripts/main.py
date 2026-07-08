import pandas as pd

from improve_email import improve_email
from metrics import compute_metrics
from pull_data import ensure_data
import classifier

MAX_PASSES = 10


def make_composite_email(row: dict) -> str:
    return (
        f"From: {row['sender']}\n"
        f"To: {row['receiver']}\n"
        f"Date: {row['date']}\n"
        f"Subject: {row['subject']}\n"
        f"\n"
        f"{row['body']}"
    )


def load_dataframes() -> dict[str, pd.DataFrame]:
    data_dir = ensure_data()
    names = ["CEAS_08", "Nazario", "Nigerian_Fraud", "SpamAssasin"]
    return {name: pd.read_csv(data_dir / f"{name}.csv") for name in names}


def main():
    dfs = load_dataframes()
    results = []

    for name, df in dfs.items():
        print(f"\n=== {name} ({len(df)} emails) ===")
        for _, row in df.iterrows():
            original_email = make_composite_email(row.to_dict())
            email = original_email
            passes = 0

            for _ in range(MAX_PASSES):
                if classifier.classify(email) != "1:phishing":
                    break
                passes += 1
                improved = improve_email(email)
                metrics = compute_metrics(email, improved)
                success = classifier.classify(improved) != "1:phishing"
                results.append({
                    "source": name,
                    "pass": passes,
                    "original": original_email,
                    "improved": improved,
                    "success": success,
                    **metrics,
                })
                email = improved

            print(f"  passes={passes} {'OK' if passes < MAX_PASSES else 'FAILED'}")

    df_results = pd.DataFrame(results)
    print(f"\n=== SUMMARY ===")
    print(f"Total improvements: {len(df_results)}")
    if not df_results.empty:
        print(df_results.describe())


if __name__ == "__main__":
    main()
