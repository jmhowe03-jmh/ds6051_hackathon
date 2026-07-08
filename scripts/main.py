import pandas as pd

from classifier import PhishingClassifier
from improve_email import improve_email
from metrics import compute_metrics, rule_compliance
from pathlib import Path

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
    data_dir = Path(__file__).resolve().parent.parent / "results"
    names = ["CEAS_08", "Nazario", "Nigerian_Fraud", "SpamAssasin"]
    return {name: pd.read_csv(data_dir / f"{name}_classified.csv") for name in names}


def main():
    clf = PhishingClassifier()
    dfs = load_dataframes()
    results = []

    for name, df in dfs.items():
        print(f"\n=== {name} ({len(df)} emails) ===")
        for _, row in df.iterrows():
            original_email = make_composite_email(row.to_dict())
            email = original_email
            passes = 0

            subject = row["subject"]
            body = row["body"]
            sender = row["sender"]
            receiver = row["receiver"]
            date = row["date"]

            for _ in range(MAX_PASSES):
                print("Classifying... ", end="", flush=True)
                if clf.classify(subject=subject, body=body, sender=sender, receiver=receiver, date=date) != "phishing":
                    break
                passes += 1
                improved = improve_email(email)
                print("metrics... ", end="", flush=True)
                metrics = compute_metrics(email, improved)
                compliance = rule_compliance(original_email, improved)
                success = clf.classify(subject=subject, body=improved, sender=sender, receiver=receiver, date=date) != "phishing"
                results.append({
                    "source": name,
                    "pass": passes,
                    "original": original_email,
                    "improved": improved,
                    "success": success,
                    **metrics,
                    **compliance,
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
