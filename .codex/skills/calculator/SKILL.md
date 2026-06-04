---
name: calculator
description: Use when a task requires precise arithmetic, numeric expression evaluation, percentages, powers, roots, rounding, or safe math checks that should not rely on mental calculation.
---

# Calculator

## Overview

Use this project skill for deterministic arithmetic instead of estimating with the model. It provides a small safe expression evaluator that accepts only numeric expressions and a short allowlist of math functions.

## When to Use

- Exact arithmetic, percentages, ratios, powers, modulo, or roots.
- Verifying a number produced by an LLM, document, spreadsheet, or code snippet.
- Any calculation where a one-digit error would change the answer.

Do not use this skill for symbolic algebra, statistics packages, file processing, network calls, or arbitrary Python execution.

## Quick Start

Run the bundled script from the repository root:

```bash
python .codex/skills/calculator/scripts/calculate.py "1234 * 5678"
```

Supported syntax:

- Numeric constants: `1`, `3.14`
- Operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Unary signs: `+1`, `-1`
- Functions: `abs`, `ceil`, `floor`, `log`, `round`, `sqrt`

## Output

Successful calculations print:

```text
计算结果: 1234 * 5678 = 7006652
```

Rejected or invalid expressions print `计算错误: ...` and exit with status `1`.

## Safety Boundary

The script parses expressions with Python `ast` and evaluates only whitelisted node types. It rejects imports, attributes, assignments, arbitrary names, file access, networking, and dynamic execution.
