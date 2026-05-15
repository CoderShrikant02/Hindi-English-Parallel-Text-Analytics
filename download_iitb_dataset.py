import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download cfilt/iitb-english-hindi dataset and optionally save a sample as CSV."
    )
    parser.add_argument(
        "--split",
        default="train[:5000]",
        help="Dataset split expression, e.g. train[:5000], validation[:1000], test",
    )
    parser.add_argument(
        "--out",
        default="iitb_sample.csv",
        help="Output CSV file path for saved sample rows.",
    )
    args = parser.parse_args()

    print("Loading dataset: cfilt/iitb-english-hindi")
    ds = load_dataset("cfilt/iitb-english-hindi", split=args.split)
    print(ds)

    out_path = Path(args.out)
    rows = []
    for item in ds:
        tr = item.get("translation", {})
        rows.append({"en": tr.get("en", ""), "hi": tr.get("hi", "")})

    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Saved {len(df)} rows to: {out_path.resolve()}")
    else:
        print("No rows to save.")


if __name__ == "__main__":
    main()
