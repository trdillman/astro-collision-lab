"""Depth must reach the real tree config, restart identity, and validation gate."""
import dataclasses

import pytest

from astro_collision_lab.simulation import BodySpec, SimulationConfig
from astro_collision_lab.interaction import physical_signature, equilibrium_signature


def test_default_depth_preserves_the_existing_solver():
    assert SimulationConfig().tree_max_depth == 16


@pytest.mark.parametrize('depth', [1, 9, 12, 16])
def test_selected_depth_reaches_real_vendor_config_with_matching_stack(depth):
    from astro_collision_lab.simulation import galaxy_tree_config

    config = SimulationConfig(particle_count=400000, tree_max_depth=depth)
    tree = galaxy_tree_config(config)
    tree.validate()
    assert tree.max_depth == depth
    assert tree.stack_capacity == 1 + 7 * depth
    assert tree.count == tree.capacity == config.particle_count
    assert tree.theta == config.theta
    assert tree.softening == config.softening


@pytest.mark.parametrize('depth', [0, 17, -1, 9.5, True])
def test_invalid_depth_is_rejected_before_sampling_or_gpu_work(depth):
    config = dataclasses.replace(SimulationConfig(), tree_max_depth=depth)
    with pytest.raises(ValueError, match='tree_max_depth'):
        config.validate()


def test_particle_limit_follows_selected_depth_and_fails_before_sampling():
    SimulationConfig(particle_count=800000, tree_max_depth=9).validate()
    with pytest.raises(ValueError, match='800000.*16.*12800001.*9000000'):
        SimulationConfig(particle_count=800000, tree_max_depth=16).validate()


def test_compact_storage_admits_the_same_count_without_lowering_depth():
    from astro_collision_lab.simulation import galaxy_tree_config

    config = SimulationConfig(particle_count=800000, tree_max_depth=16, tree_storage='COMPACT')
    config.validate()
    tree = galaxy_tree_config(config)
    assert tree.max_depth == 16
    assert tree.count == 800000
    assert tree.node_capacity > 0
    assert tree.node_capacity < 9000000


def test_unknown_storage_is_refused_and_storage_edits_invalidate_identity():
    with pytest.raises(ValueError, match='tree_storage'):
        SimulationConfig(tree_storage='IGNORE_OVERFLOW').validate()
    old = SimulationConfig(particle_count=32)
    new = dataclasses.replace(old, tree_storage='COMPACT')
    bodies = [BodySpec('Galaxy', particle_count=32)]
    assert physical_signature(old, bodies) != physical_signature(new, bodies)


def test_depth_edit_invalidates_playback_and_isolation_identity():
    old = SimulationConfig(particle_count=32)
    new = dataclasses.replace(old, tree_max_depth=9)
    bodies = [BodySpec('Galaxy', particle_count=32)]
    assert physical_signature(old, bodies) != physical_signature(new, bodies)
    assert equilibrium_signature(old, bodies) != equilibrium_signature(new, bodies)
    assert SimulationConfig(**dataclasses.asdict(new)) == new
