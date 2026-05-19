"""BrainDB Python integration tests."""

import pytest


def test_import():
    """Verify the braindb package can be imported."""
    import braindb
    assert hasattr(braindb, "BrainDB")
    assert hasattr(braindb, "Simulation")
    assert hasattr(braindb, "SpikeLog")


def test_braindb_open(tmp_path):
    """Open a .braindb file created by the CLI."""
    db_path = tmp_path / "test.braindb"
    # This test requires a pre-built .braindb file.
    # In CI, run `braindb-cli build -o test.braindb` first.
    if not db_path.exists():
        pytest.skip("No test.braindb file available")

    from braindb import BrainDB
    db = BrainDB(str(db_path))
    assert db.neuron_count() > 0


def test_simulation_run(tmp_path):
    """Run a simulation on a .braindb file."""
    db_path = tmp_path / "test.braindb"
    if not db_path.exists():
        pytest.skip("No test.braindb file available")

    from braindb import Simulation
    sim = Simulation(str(db_path))
    sim.set_neuron_input(0, 30.0)
    sim.run(1000)  # 100 ms
    v = sim.get_neuron_voltage(0)
    assert -120.0 < v < 100.0, f"Voltage {v} out of physiological range"
