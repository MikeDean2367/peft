# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys

import pytest

from .test_cortex_consolidation_example import EXAMPLES, scoring


def test_cortex_evaluation_defaults_to_generation(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sys, "argv", ["eval.py", "--output-file", str(tmp_path / "responses.json")])
    monkeypatch.setattr(scoring, "evaluate", lambda args: calls.append("generate"))

    def forbidden(*args):
        pytest.fail("Generation must not check execution dependencies or run model-generated code.")

    monkeypatch.setattr(scoring, "check_execution_environment", forbidden)
    monkeypatch.setattr(scoring, "score_report", forbidden)
    scoring.main()
    assert calls == ["generate"]
    args = scoring.parse_args(["--output-file", str(tmp_path / "other.json")])
    assert not args.execute


@pytest.mark.parametrize(
    "options",
    [
        ["--generate-only"],
        ["--execute"],
        ["--report", "responses.json"],
        ["--execute", "--generate-only"],
        ["--report", "responses.json", "--execute", "--generate-only"],
    ],
)
def test_cortex_evaluation_requires_explicit_scoring(options, tmp_path):
    with pytest.raises(SystemExit) as error:
        scoring.parse_args([*options, "--output-file", str(tmp_path / "scored.json")])
    assert error.value.code == 2


def test_cortex_explicit_report_scoring_does_not_load_model(monkeypatch, tmp_path):
    source, output = tmp_path / "responses.json", tmp_path / "scored.json"
    source.write_text(json.dumps({"summary": {}, "probes": []}))
    calls = []
    monkeypatch.setattr(sys, "argv", ["eval.py", "--report", str(source), "--execute", "--output-file", str(output)])
    monkeypatch.setattr(scoring, "check_execution_environment", lambda: calls.append("check"))
    monkeypatch.setattr(scoring, "score_report", lambda report, timeout: {**report, "scored": True})
    monkeypatch.setattr(scoring, "evaluate", lambda args: pytest.fail("Report scoring must not load a model."))
    scoring.main()
    assert calls == ["check"] and json.loads(output.read_text())["scored"]


@pytest.mark.parametrize("include_retain", [False, True])
def test_cortex_code_scores_are_grouped_by_split(monkeypatch, include_retain):
    monkeypatch.setattr(scoring, "run_tests", lambda response, *args: {"passed": response == "correct"})

    def task(name, split, answer, overlap=False):
        return {
            "id": name,
            "split": split,
            "format": "python",
            "tests": "tests",
            "overlaps_training_question": overlap,
            "base": {"response": answer},
        }

    report = {
        "summary": {"retain": {"num_probes": 2, "base": {"exact_match": 0.5}}} if include_retain else {},
        "probes": [
            task("target", "target", "correct"),
            task("target_overlap", "target", "wrong", overlap=True),
        ],
    }
    if include_retain:
        report["probes"].extend(
            [task("retain", "retain", "wrong"), task("retain_overlap", "retain", "correct", overlap=True)]
        )
    result = scoring.score_report(report, 1)
    assert set(result["summary"]) == ({"target", "retain"} if include_retain else {"target"})
    assert result["summary"]["target"]["base"]["pass_rate"] == 1
    assert result["summary"]["target"]["num_tasks"] == 1
    if include_retain:
        assert result["summary"]["retain"]["base"]["pass_rate"] == 0
        assert result["summary"]["retain"]["num_tasks"] == 1
        assert result["summary"]["retain"]["base"]["exact_match"] == 0.5
        assert result["summary"]["retain"]["num_probes"] == 2
    assert result["execution"]["backend"] == "local_subprocess"


@pytest.mark.parametrize("split", [None, "train", ""])
def test_cortex_code_scoring_rejects_missing_or_invalid_split(monkeypatch, split):
    monkeypatch.setattr(scoring, "run_tests", lambda *args: pytest.fail("Validate the report before executing code."))
    report = {
        "summary": {},
        "probes": [{"id": "code", "format": "python", "tests": "tests", "split": split, "base": {"response": "code"}}],
    }
    with pytest.raises(ValueError, match="split=target/retain"):
        scoring.score_report(report, 1)


REFERENCE_PROGRAMS = {
    "seis_01": "from __future__ import annotations\n"
    "\n"
    "import seismic_linter\n"
    "\n"
    "\n"
    "def t003_line(source: str) -> int | None:\n"
    '    violations = seismic_linter.analyze_code(source, filename="x.py")\n'
    "    for v in violations:\n"
    '        if v.rule_id == "T003":\n'
    "            return int(v.lineno)\n"
    "    return None\n",
    "seis_02": "from __future__ import annotations\n"
    "\n"
    "import seismic_linter\n"
    "\n"
    "\n"
    "def rule_ids(source: str) -> list[str]:\n"
    '    violations = seismic_linter.analyze_code(source, filename="x.py")\n'
    "    return sorted({v.rule_id for v in violations})\n",
    "seis_03": "from __future__ import annotations\n"
    "\n"
    "import seismic_linter\n"
    "\n"
    "\n"
    "def is_clean(source: str) -> bool:\n"
    '    return len(seismic_linter.analyze_code(source, filename="x.py")) == 0\n',
    "seis_04": "from __future__ import annotations\n"
    "\n"
    "from pathlib import Path\n"
    "\n"
    "import seismic_linter\n"
    "\n"
    "\n"
    "def count_violations_in_file(file_path: str) -> int:\n"
    "    v = seismic_linter.analyze_file(Path(file_path))\n"
    "    return int(len(v))\n",
    "seis_05": "from __future__ import annotations\n"
    "\n"
    "import pandas as pd\n"
    "import seismic_linter\n"
    "\n"
    "\n"
    "def split_ok(train_times: list[int], test_times: list[int]) -> bool:\n"
    '    train = pd.DataFrame({"time": list(train_times)})\n'
    '    test = pd.DataFrame({"time": list(test_times)})\n'
    "    try:\n"
    "        seismic_linter.validate_split_integrity(train, test)\n"
    "        return True\n"
    "    except seismic_linter.TemporalCausalityError:\n"
    "        return False\n",
    "seis_06": "from __future__ import annotations\n"
    "\n"
    "import pandas as pd\n"
    "import seismic_linter\n"
    "\n"
    "\n"
    "def lint_and_split_summary(source: str, train_times: list[int], test_times: list[int]) -> "
    "dict:\n"
    '    v = seismic_linter.analyze_code(source, filename="x.py")\n'
    '    train = pd.DataFrame({"time": list(train_times)})\n'
    '    test = pd.DataFrame({"time": list(test_times)})\n'
    "    ok = True\n"
    "    try:\n"
    "        seismic_linter.validate_split_integrity(train, test)\n"
    "    except seismic_linter.TemporalCausalityError:\n"
    "        ok = False\n"
    '    return {"violations": int(len(v)), "split_ok": bool(ok)}\n',
    "seis_07": "from __future__ import annotations\n"
    "\n"
    "import pandas as pd\n"
    "import seismic_linter\n"
    "\n"
    "\n"
    "def make_monotonic_df(times: list[int]):\n"
    "    times = list(times)\n"
    "\n"
    '    @seismic_linter.verify_monotonicity(time_col="time")\n'
    "    def _load():\n"
    '        return pd.DataFrame({"time": sorted(times)})\n'
    "\n"
    "    return _load()\n",
    "seis_08": "from __future__ import annotations\n"
    "\n"
    "import seismic_linter\n"
    "\n"
    "\n"
    "def count_violations(source: str) -> int:\n"
    '    v = seismic_linter.analyze_code(source, filename="x.py")\n'
    "    return int(len(v))\n",
    "seis_09": "from __future__ import annotations\n"
    "\n"
    "import pandas as pd\n"
    "import seismic_linter\n"
    "\n"
    "\n"
    "def split_ok_debug(train_times: list[int], test_times: list[int]) -> bool:\n"
    '    train = pd.DataFrame({"time": list(train_times)})\n'
    '    test = pd.DataFrame({"time": list(test_times)})\n'
    "    try:\n"
    "        seismic_linter.validate_split_integrity(train, test)\n"
    "        return True\n"
    "    except seismic_linter.TemporalCausalityError:\n"
    "        return False\n",
    "seis_10": "from __future__ import annotations\n"
    "\n"
    "import pandas as pd\n"
    "import seismic_linter\n"
    "\n"
    "\n"
    '@seismic_linter.verify_monotonicity(time_col="time")\n'
    "def load_df_sorted():\n"
    '    return pd.DataFrame({"time": [1, 2, 3]})\n',
}


def bundled_task_tests(sample_id):
    return next(
        row["tests"]
        for row in scoring.read_jsonl(EXAMPLES / "consolidate" / "data.jsonl")
        if row.get("id") == sample_id
    )


@pytest.mark.parametrize("sample_id", sorted(REFERENCE_PROGRAMS))
def test_cortex_bundled_scoring_accepts_real_api_reference(sample_id):
    package = pytest.importorskip("seismic_linter")
    assert package is not None
    scoring.check_execution_environment()
    result = scoring.run_tests(
        "```python\n" + REFERENCE_PROGRAMS[sample_id] + "\n```", bundled_task_tests(sample_id), 30
    )
    assert result["passed"], result


@pytest.mark.parametrize(
    "sample_id,program",
    [
        ("seis_02", "def rule_ids(source): return ['T003']"),
        ("seis_04", "def count_violations_in_file(file_path): return 1"),
        (
            "seis_06",
            "def lint_and_split_summary(source, train_times, test_times): return {'violations': 1, 'split_ok': False}",
        ),
        ("seis_07", "import pandas as pd\ndef make_monotonic_df(times): return pd.DataFrame({'time': [1, 2, 3]})"),
        ("seis_08", "def count_violations(source): return 1"),
        ("seis_10", "import pandas as pd\ndef load_df_sorted(): return pd.DataFrame({'time': [1, 2, 3]})"),
    ],
)
def test_cortex_bundled_scoring_rejects_constant_or_unvalidated_answers(sample_id, program):
    pytest.importorskip("seismic_linter")
    scoring.check_execution_environment()
    result = scoring.run_tests("```python\n" + program + "\n```", bundled_task_tests(sample_id), 30)
    assert not result["passed"] and not result["timed_out"], result
