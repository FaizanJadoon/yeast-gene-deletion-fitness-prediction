"""
Faizan Paper 3
Step 10: Public dataset availability audit.

Checks official dataset sources without downloading
large genomic or phenotype datasets.

Output:
data/metadata/source_access_audit.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "metadata" / "source_access_audit.json"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

SOURCES = [
    {
        "name": "Yeast Phenome Downloads",
        "category": "harmonized",
        "method": "GET",
        "url": "https://www.yeastphenome.org/downloads/",
    },
    {
        "name": "Yeast Phenome Bulk Dataset",
        "category": "harmonized",
        "method": "HEAD",
        "url": (
            "https://storage.googleapis.com/"
            "yeastphenome-downloads/"
            "yp_haphom_20221025.tar.gz"
        ),
    },
    {
        "name": "Yeast Phenome GitHub",
        "category": "provenance",
        "method": "GET",
        "url": "https://api.github.com/repos/yeastphenome/yp-data",
    },
    {
        "name": "Piggott 2011 Fermentation",
        "category": "food",
        "method": "GET",
        "url": "https://www.ebi.ac.uk/biostudies/studies/E-MEXP-3332",
    },
    {
        "name": "Novo 2013 Fermentation",
        "category": "food",
        "method": "GET",
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE46145",
    },
    {
        "name": "Chemical Genomics 2022",
        "category": "chemical",
        "method": "GET",
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186866",
    },
]


def check_source(session, source):
    result = dict(source)

    try:
        method = source["method"]

        with session.request(
            method,
            source["url"],
            timeout=35,
            allow_redirects=True,
            stream=True,
        ) as response:
            result["status_code"] = response.status_code
            result["accessible"] = response.ok
            result["final_url"] = response.url
            result["content_type"] = response.headers.get(
                "Content-Type"
            )
            result["content_length"] = response.headers.get(
                "Content-Length"
            )
            result["etag"] = response.headers.get("ETag")
            result["last_modified"] = response.headers.get(
                "Last-Modified"
            )
            result["error"] = None

    except requests.RequestException as exc:
        result["accessible"] = False
        result["status_code"] = None
        result["error"] = str(exc)

    return result


def main():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Faizan-Paper3-Research-Audit/1.0",
        "Accept": "text/html,application/json,*/*",
    })

    report = {
        "project": "Faizan Paper 3",
        "step": 10,
        "checked_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "purpose": "Dataset source availability audit",
        "checks": [],
    }

    print("\nFAIZAN PAPER 3 - DATASET SOURCE AUDIT")
    print("=" * 55)

    for source in SOURCES:
        result = check_source(session, source)
        report["checks"].append(result)

        status = (
            "ACCESSIBLE"
            if result["accessible"]
            else "CHECK REQUIRED"
        )

        print(f"\n{source['name']}")
        print(f"Status: {status}")
        print(f"HTTP: {result['status_code']}")

        if result.get("error"):
            print(f"Error: {result['error']}")

    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    accessible = sum(
        item["accessible"] for item in report["checks"]
    )

    print("\n" + "=" * 55)
    print(f"Accessible sources: {accessible}/{len(SOURCES)}")
    print(f"Audit saved: {OUTPUT}")
    print(
        "NOTE: Source accessibility does not prove "
        "dataset compatibility or scientific suitability."
    )


if __name__ == "__main__":
    main()