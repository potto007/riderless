"""SemIf-style JSONL interchange: one row, one Choice question, 2 to 16 options."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Self

from pydantic import Field, model_validator

from systemone.api.schema import (
    ChoiceQuestion,
    Entry,
    State,
    StrictModel,
    SystemOneRequest,
)


class SemIfOption(StrictModel):
    id: str = Field(min_length=1)
    description: Entry


class SemIfRow(StrictModel):
    id: str = Field(min_length=1)
    state: State
    question: Entry
    options: list[SemIfOption] = Field(min_length=2, max_length=16)

    @model_validator(mode="after")
    def _unique_options(self) -> Self:
        ids = [option.id for option in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("SemIf option ids must be unique")
        return self


def import_semif_row(value: object) -> SystemOneRequest:
    try:
        row = SemIfRow.model_validate(value)
    except ValueError as error:
        raise ValueError("SemIf rows require 2 to 16 valid options") from error
    return SystemOneRequest(
        state=row.state,
        questions={
            row.id: ChoiceQuestion(
                type="choice",
                instructions=row.question,
                criteria={option.id: option.description for option in row.options},
            )
        },
    )


def export_semif_row(request: SystemOneRequest) -> SemIfRow:
    if len(request.questions) != 1:
        raise ValueError("SemIf export requires exactly one question")
    question_id, question = next(iter(request.questions.items()))
    if not isinstance(question, ChoiceQuestion):
        raise TypeError("SemIf export requires a Choice question")
    if not 2 <= len(question.criteria) <= 16:
        raise ValueError("SemIf export requires 2 to 16 options")
    return SemIfRow(
        id=question_id,
        state=request.state,
        question=question.instructions,
        options=[
            SemIfOption(id=option_id, description=description)
            for option_id, description in question.criteria.items()
        ],
    )


def _load_jsonl(path: Path) -> list[Any]:
    values: list[Any] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at line {line_number}") from error
    if not values:
        raise ValueError("input JSONL is empty")
    return values


def _write_jsonl(path: Path, values: list[StrictModel]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(value.model_dump_json(exclude_none=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("to-api", "from-api"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = _load_jsonl(args.input)
        if args.direction == "to-api":
            converted: list[StrictModel] = [import_semif_row(row) for row in raw]
        else:
            converted = [
                export_semif_row(SystemOneRequest.model_validate(row)) for row in raw
            ]
        _write_jsonl(args.output, converted)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
