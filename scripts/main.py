import pandas as pd

from improve_email import improve_email
from pull_data import ensure_data
import classifier


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
    history = []

    for name, df in dfs.items():
        print(f"\n=== {name} ({len(df)} emails) ===")
        for _, row in df.iterrows():
            email = make_composite_email(row.to_dict())
            passes = 0

            while classifier.classify(email) == "1:phishing":
                passes += 1
                email = improve_email(email)
                history.append({
                    "source": name,
                    "pass": passes,
                    "email": email,
                })

            if passes > 0:
                print(f"  Improved in {passes} pass(es)")


if __name__ == "__main__":
    main()
