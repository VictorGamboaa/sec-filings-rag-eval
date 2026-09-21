"""Build inputs/universe.csv by joining tickers against SEC's official mapping.

    python -m tools.build_universe
    python -m tools.build_universe --dry-run
    python -m tools.build_universe --from-cache
    python -m tools.build_universe --set universe.tickers='["AAPL","MSFT"]'

WHY THIS IS A FETCH AND NOT A LOOKUP
====================================

A CIK typed from memory is a generated value: it has no source and no audit
standing, and a wrong one fails in the worst possible way -- silently, by
building a perfectly valid corpus of the wrong company's filings. Every CIK in
the universe therefore comes from SEC's own company_tickers.json, fetched at
build time, and the fetched file is kept at ``universe.cache_path`` as the audit
trail for the join.

The join is exact and case-insensitive on ticker. Zero matches or more than one
match is a hard failure naming the ticker -- never a nearest-match guess, and
never a silent drop that would leave a short universe looking complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tools.config import ConfigError, add_config_args, load_config
from tools.edgar import EdgarClient, EdgarError, cik10, write_universe
from tools.runlog import RunLog

__all__ = ["build_universe", "parse_company_tickers"]


def parse_company_tickers(payload: Any) -> list[dict[str, str]]:
    """Validate and flatten SEC's company_tickers.json.

    The documented shape is a mapping of arbitrary string keys to records::

        {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}

    Validated rather than assumed: if SEC changes the shape, this must fail
    loudly at the join, not produce an empty or partial universe that later
    stages would treat as the real corpus.
    """
    if not isinstance(payload, dict) or not payload:
        raise EdgarError(
            f"company_tickers.json was not a non-empty JSON object "
            f"(got {type(payload).__name__}). SEC may have changed the format; "
            f"the join cannot proceed without a verified mapping."
        )

    records: list[dict[str, str]] = []
    for key, row in payload.items():
        if not isinstance(row, dict):
            raise EdgarError(
                f"company_tickers.json entry {key!r} is a "
                f"{type(row).__name__}, expected an object with cik_str/ticker/title."
            )
        missing = [f for f in ("cik_str", "ticker", "title") if f not in row]
        if missing:
            raise EdgarError(
                f"company_tickers.json entry {key!r} is missing {missing}; "
                f"got keys {sorted(row)}."
            )
        records.append(
            {
                "ticker": str(row["ticker"]).strip().upper(),
                "cik": cik10(row["cik_str"]),
                "company_name": str(row["title"]).strip(),
            }
        )
    return records


def _join(records: list[dict[str, str]], wanted: list[str]) -> list[dict[str, str]]:
    """Exact case-insensitive ticker join. Ambiguity and absence both fail hard."""
    by_ticker: dict[str, list[dict[str, str]]] = {}
    for rec in records:
        by_ticker.setdefault(rec["ticker"], []).append(rec)

    resolved: list[dict[str, str]] = []
    problems: list[str] = []
    for raw in wanted:
        ticker = str(raw).strip().upper()
        matches = by_ticker.get(ticker, [])
        if not matches:
            problems.append(
                f"{ticker}: no match in SEC's company_tickers.json "
                f"(the file lists {len(records)} tickers)"
            )
        elif len(matches) > 1:
            ciks = ", ".join(sorted({m['cik'] for m in matches}))
            problems.append(
                f"{ticker}: {len(matches)} matches with CIKs [{ciks}]; "
                f"ambiguous, so no CIK can be chosen without guessing"
            )
        else:
            resolved.append(dict(matches[0]))

    if problems:
        raise EdgarError(
            "ticker join failed; universe.csv not written:\n  "
            + "\n  ".join(problems)
        )
    return resolved


def build_universe(
    config_path: str,
    overrides: list[str],
    *,
    dry_run: bool = False,
    from_cache: bool = False,
) -> int:
    config = load_config(config_path, overrides=overrides)

    url = config.get("universe.tickers_url")
    out_path = Path(config.get("universe.out_path"))
    cache_path = Path(config.get("universe.cache_path"))
    wanted = list(config.get("universe.tickers"))

    if dry_run:
        print("DRY RUN -- no request made, no file written.")
        print(f"  would GET   {url}")
        print(f"  would cache {cache_path}")
        print(f"  would write {out_path}  (columns: ticker,cik,company_name)")
        print(f"  tickers     {', '.join(wanted)}")
        return 0

    with RunLog(config) as log:
        with log.stage("build_universe") as stage:
            stage.note(tickers=wanted, source_url=url, from_cache=from_cache)
            try:
                if from_cache:
                    if not cache_path.is_file():
                        raise EdgarError(
                            f"--from-cache given but {cache_path} does not exist. "
                            f"Run once without it to fetch the mapping."
                        )
                    raw = cache_path.read_bytes()
                    stage.bytes_in(len(raw))
                    payload = json.loads(raw.decode("utf-8"))
                else:
                    with EdgarClient(config, stage) as client:
                        response = client.get(url)
                        raw = response.content
                        payload = json.loads(raw.decode("utf-8"))
                        # Cache before parsing succeeds or fails: the bytes SEC
                        # actually served are the audit trail for this join, and
                        # they are most worth having when the parse went wrong.
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_path.write_bytes(raw)
                        stage.bytes_out(len(raw))
                    stage.note(throttle_wait_sec=client.throttle_wait_sec)

                records = parse_company_tickers(payload)
                stage.note(source_tickers=len(records))

                rows = _join(records, wanted)
                size = write_universe(out_path, rows)
                stage.count(len(rows))
                stage.bytes_out(size)
                stage.note(universe_rows=len(rows), out_path=str(out_path))
            except (EdgarError, ValueError) as exc:
                stage.error(exc, context={"url": url})
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

    print(f"Wrote {out_path} ({len(rows)} rows) from {url}")
    print(f"  source kept for audit at {cache_path}")
    print()
    print(f"  {'ticker':<8} {'cik':<12} company_name")
    for row in rows:
        print(f"  {row['ticker']:<8} {row['cik']:<12} {row['company_name']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.build_universe",
        description="Build inputs/universe.csv from SEC's official ticker->CIK file.",
    )
    add_config_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be requested and written; make no request",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="re-join using the previously fetched mapping; make no request",
    )
    args = parser.parse_args(argv)

    try:
        return build_universe(
            args.config,
            args.overrides,
            dry_run=args.dry_run,
            from_cache=args.from_cache,
        )
    except (ConfigError, EdgarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
