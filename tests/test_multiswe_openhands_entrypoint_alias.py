import pathlib

from verifiers.integrations.multiswebench.runner import OpenHandsRunner


def test_openhands_entrypoint_alias_precedence():
    # Use any existing file as dataset_file to satisfy constructor checks; no harness invocation.
    dataset_file = pathlib.Path(__file__).as_posix()
    r = OpenHandsRunner(
        dataset_file=dataset_file,
        output_dir="./msb_runs_test",
        openhands_entrypoint="mopenhands.preferred",
        entrypoint_module="mopenhands.legacy",
    )
    # The new name must take precedence over the legacy alias.
    assert getattr(r, "entrypoint_module") == "mopenhands.preferred"

