import pytest

from osmose.scenarios import Scenario, ScenarioManager


@pytest.fixture
def manager(tmp_path):
    return ScenarioManager(tmp_path / "scenarios")


@pytest.fixture
def sample_scenario():
    return Scenario(
        name="baltic_smelt_baseline",
        description="Baseline Baltic smelt configuration",
        config={
            "simulation.nspecies": "3",
            "species.name.sp0": "Smelt",
            "species.linf.sp0": "25.0",
        },
        tags=["baseline", "baltic"],
    )


def test_save_and_load(manager, sample_scenario):
    manager.save(sample_scenario)
    loaded = manager.load("baltic_smelt_baseline")
    assert loaded.name == "baltic_smelt_baseline"
    assert loaded.config["simulation.nspecies"] == "3"
    assert loaded.tags == ["baseline", "baltic"]


def test_save_creates_directory(manager, sample_scenario):
    path = manager.save(sample_scenario)
    assert path.exists()
    assert (path / "scenario.json").exists()


def test_list_scenarios(manager, sample_scenario):
    manager.save(sample_scenario)
    manager.save(Scenario(name="another", config={"x": "1"}))
    listing = manager.list_scenarios()
    assert len(listing) == 2
    names = [s["name"] for s in listing]
    assert "baltic_smelt_baseline" in names
    assert "another" in names


def test_delete_scenario(manager, sample_scenario):
    manager.save(sample_scenario)
    assert len(manager.list_scenarios()) == 1
    manager.delete("baltic_smelt_baseline")
    assert len(manager.list_scenarios()) == 0


def test_compare_scenarios(manager):
    manager.save(Scenario(name="a", config={"x": "1", "y": "2", "z": "3"}))
    manager.save(Scenario(name="b", config={"x": "1", "y": "99", "w": "4"}))
    diffs = manager.compare("a", "b")
    keys = [d.key for d in diffs]
    assert "y" in keys  # different value
    assert "z" in keys  # only in a
    assert "w" in keys  # only in b
    assert "x" not in keys  # same value


def test_compare_finds_value_changes(manager):
    manager.save(Scenario(name="a", config={"species.linf.sp0": "25.0"}))
    manager.save(Scenario(name="b", config={"species.linf.sp0": "30.0"}))
    diffs = manager.compare("a", "b")
    assert len(diffs) == 1
    assert diffs[0].value_a == "25.0"
    assert diffs[0].value_b == "30.0"


def test_fork_scenario(manager, sample_scenario):
    manager.save(sample_scenario)
    forked = manager.fork("baltic_smelt_baseline", "high_fishing", "High fishing scenario")
    assert forked.name == "high_fishing"
    assert forked.parent_scenario == "baltic_smelt_baseline"
    assert forked.config == sample_scenario.config
    # Verify it was saved
    loaded = manager.load("high_fishing")
    assert loaded.name == "high_fishing"


def test_fork_is_independent(manager, sample_scenario):
    manager.save(sample_scenario)
    forked = manager.fork("baltic_smelt_baseline", "variant")
    forked.config["species.linf.sp0"] = "999"
    manager.save(forked)
    # Original should be unchanged
    original = manager.load("baltic_smelt_baseline")
    assert original.config["species.linf.sp0"] == "25.0"


def test_scenario_timestamps(manager, sample_scenario):
    manager.save(sample_scenario)
    loaded = manager.load("baltic_smelt_baseline")
    assert loaded.created_at
    assert loaded.modified_at


def test_load_nonexistent_raises(manager):
    with pytest.raises(FileNotFoundError):
        manager.load("nonexistent")
