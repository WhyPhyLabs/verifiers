import json
import tempfile
from pathlib import Path

from verifiers.integrations.multiswebench.runner import _report_indicates_resolved


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_report_shape_resolved_instances_list_final():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_json(d / "final_report.json", {"resolved_instances": ["org/repo:pr-123"]})
        resolved, info = _report_indicates_resolved(d)
        assert resolved is True
        assert info.get("final_report") is True


def test_report_shape_instances_map_legacy():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_json(d / "report.json", {"instances": {"org__repo-123": {"resolved": True}}})
        resolved, info = _report_indicates_resolved(d)
        assert resolved is True
        assert info.get("legacy_report") is True

