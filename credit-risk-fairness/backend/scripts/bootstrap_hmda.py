from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlretrieve


API_ROOT = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a public HMDA loan-level CSV from the FFIEC Data Browser.")
    parser.add_argument("--year", default="2023")
    parser.add_argument("--state", default=None, help="Two-letter state code, e.g. IL.")
    parser.add_argument("--county", default="17187", help="Five-digit county FIPS. Default is Warren County, IL.")
    parser.add_argument("--actions", default="1,3", help="HMDA action_taken codes. 1=originated, 3=denied.")
    parser.add_argument("--out", default="backend/data/hmda_live.csv")
    args = parser.parse_args()

    params = {"years": args.year, "actions_taken": args.actions}
    if args.county:
        params["counties"] = args.county
    elif args.state:
        params["states"] = args.state
    else:
        raise SystemExit("Provide --county or --state to keep the download bounded.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API_ROOT}?{urlencode(params)}"
    print(f"Downloading {url}")
    urlretrieve(url, out)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()

