#!/usr/bin/env python3
"""Check that recorded solution sources still match solutions/*/submissions.json.

A source file named by a submission record is frozen: its SHA-256 must stay
equal to the recorded `hash.repositoryCode`. This catches a formatter, a lint
fix or an edit that would silently detach a file from its platform results.

Standard library only, Python 3.9+, no network and no credentials.
"""

import hashlib
import json
import sys
from pathlib import Path


def check(root):
    problems = []
    checked = set()
    reported = set()
    for record_path in sorted(root.glob("solutions/*/submissions.json")):
        try:
            submissions = json.loads(record_path.read_text(encoding="utf-8"))[
                "submissions"
            ]
        except (ValueError, KeyError, TypeError) as error:
            problems.append(f"{record_path}: unreadable ({type(error).__name__})")
            continue
        for entry in submissions:
            name = entry.get("sourceVersion")
            expected = (entry.get("hash") or {}).get("repositoryCode")
            label = f"{record_path}: submission {str(entry.get('id'))[:8]}"
            # A record names one file in its own directory, never a path.
            if not name or Path(name).name != name or not expected:
                problems.append(f"{label} lacks sourceVersion or repositoryCode")
                continue
            source = record_path.parent / name
            if not source.is_file():
                problems.append(f"{label} names missing file {name}")
                continue
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            # Several submissions can share one file; report each mismatch once.
            if actual != expected and (source, expected) not in reported:
                reported.add((source, expected))
                problems.append(
                    f"{source}: sha256 {actual[:12]} differs from recorded "
                    f"{expected[:12]}; recorded versions are frozen, so restore "
                    "the file and put the change in a new version"
                )
            checked.add(source)
    return problems, checked


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent
    problems, checked = check(root)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"{len(checked)} recorded source file(s) match their submission records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
