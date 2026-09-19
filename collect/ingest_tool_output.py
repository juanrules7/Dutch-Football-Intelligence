"""
Moves data from the browser to disk.

The browser sandbox can't send data to localhost, so scraped batches are read
out with `javascript_tool`. Results too big to return inline get saved by the
tool to a text file (a JSON list wrapping a double-encoded string, followed by
a "(captured at origin ...)" footer). This helper unwraps that file.

Usage:
    python collect/ingest_tool_output.py <tool_result_file> <out_name> [--append]

Writes data/raw/<out_name>.json (or appends rows to data/raw/<out_name>.jsonl
when the payload has a top-level "rows" list and --append is given).
"""
import json
import os
import sys

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")


def unwrap(path):
    with open(path, encoding="utf-8") as f:
        wrapper = json.load(f)
    text = wrapper[0]["text"]
    inner, _ = json.JSONDecoder().raw_decode(text)
    return json.loads(inner) if isinstance(inner, str) else inner


def main():
    path, name = sys.argv[1], sys.argv[2]
    append = "--append" in sys.argv
    payload = unwrap(path)
    os.makedirs(RAW_DIR, exist_ok=True)
    if append and isinstance(payload, dict) and "rows" in payload:
        out = os.path.join(RAW_DIR, f"{name}.jsonl")
        with open(out, "a", encoding="utf-8") as f:
            for row in payload["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"appended {len(payload['rows'])} rows to {out}")
    else:
        if isinstance(payload, dict):
            payload.pop("pad", None)
        out = os.path.join(RAW_DIR, f"{name}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
