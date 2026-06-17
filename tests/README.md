# Tests

This folder contains isolated tests for the app without mixing test code into `codebase/`.

See [`TESTING.md`](./TESTING.md) for the full testing plan and spreadsheet-friendly tables.

## Run

```bash
python -m pytest tests -q
```

## Fast checks

```bash
python -m pytest tests/unit -q
```

```bash
python -m pytest tests/integration -q
```
