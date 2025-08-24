import pytest


def test_tb_loader_requires_dataset_when_harness_forced():
    import importlib
    env_mod = importlib.import_module("environments.vf_terminal_bench.vf_terminal_bench")
    with pytest.raises(ValueError):
        env_mod.load_environment(task_id="stub", use_harness=True, dataset_path=None)


def test_msb_loader_requires_dataset_and_entrypoint_when_openhands_forced():
    import importlib
    env_mod = importlib.import_module("environments.vf_multi_swe_bench.vf_multi_swe_bench")
    # Missing dataset_file
    with pytest.raises(ValueError):
        env_mod.load_environment(instance_id="stub", use_openhands=True, dataset_file=None, openhands_entrypoint="mopenhands.run")
    # Missing entrypoint
    with pytest.raises(ValueError):
        env_mod.load_environment(instance_id="stub", use_openhands=True, dataset_file="/tmp/dataset.jsonl", openhands_entrypoint=None)

