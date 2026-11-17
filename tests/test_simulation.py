import importlib.util
import unittest
from unittest.mock import Mock


class StandaloneSimulationContracts(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('astro_collision_lab.simulation'),
                             'independent simulation adapter is missing')
        from astro_collision_lab import simulation
        return simulation

    def test_100k_impact_is_exactly_100k_active_and_no_mirrored_slots(self):
        a=self.api();cfg=a.SimulationConfig(mode='IMPACT',particle_count=100000)
        p=a.prepare_scene([a.BodySpec(name='Earth',radius=1,mass=1,particle_count=91130),
                           a.BodySpec(name='Theia',radius=.46,mass=.15,particle_count=8870,position=(2.5,0,0))],cfg)
        self.assertEqual(len(p.bodies),100000)
        self.assertEqual(int(p.bodies.active.sum()),100000)
        self.assertEqual(len(p.impact_layout.solver_slots),100000)
        import numpy as np
        self.assertTrue(np.array_equal(p.impact_layout.source,np.arange(100000)))
        self.assertAlmostEqual(float(p.bodies.masses.sum()),1.15,places=11)

    def test_iisph_refuses_authored_spacing_mismatch_before_gpu(self):
        a=self.api()
        with self.assertRaisesRegex(ValueError,'common spacing'):
            a.prepare_scene([a.BodySpec('Target',particle_count=80000),
                a.BodySpec('Impactor',particle_count=20000,radius=.46,position=(1.6,0,0))],
                a.SimulationConfig(mode='IMPACT',particle_count=100000))

    def test_100k_galaxy_samples_authored_disk_plane_and_all_components(self):
        a=self.api();cfg=a.SimulationConfig(mode='GALAXY',particle_count=100000)
        turn=((1,0,0),(0,0,-1),(0,1,0))
        p=a.prepare_scene([a.BodySpec('A',kind='DISK',radius=1,thickness=.02,mass=.2,particle_count=35000,rotation=turn),
            a.BodySpec('AH',kind='HALO',radius=.7,mass=.8,particle_count=15000),
            a.BodySpec('B',kind='DISK',radius=1,thickness=.02,mass=.2,particle_count=35000,position=(3,0,0)),
            a.BodySpec('BH',kind='HALO',radius=.7,mass=.8,particle_count=15000,position=(3,0,0))],cfg)
        self.assertEqual(len(p.bodies),100000)
        self.assertLess(float(p.bodies.positions[:35000,1].std()),.011)
        self.assertGreater(float(p.bodies.velocities[:35000].std()),0)

    def test_count_mismatch_and_nonsphere_impact_refuse_before_gpu(self):
        a=self.api()
        with self.assertRaisesRegex(ValueError,'count'):
            a.prepare_scene([a.BodySpec('Only',particle_count=99)],a.SimulationConfig(particle_count=100))
        with self.assertRaisesRegex(ValueError,'sphere|SPHERE'):
            a.prepare_scene([a.BodySpec('Disk',kind='DISK',particle_count=100)],a.SimulationConfig(mode='IMPACT',particle_count=100))

    def test_adapter_does_not_initialize_gpu_until_explicit_initialize(self):
        a=self.api();sim=a.AstroCollisionSimulation(a.SimulationConfig())
        self.assertEqual(sim.count,0)
        self.assertIsNone(sim.runtime)
        with self.assertRaisesRegex(RuntimeError,'initializ'):sim.step()

    def test_frame_clock_counts_frames_not_tree_physical_substeps(self):
        a=self.api();sim=a.AstroCollisionSimulation(a.SimulationConfig(substeps=2))
        class BoundaryRuntime:
            step_index=0
            last_step_ms=1.0
            def step(self):self.step_index+=2
            def display_channels(self):return {'positions':object()}
        sim.runtime=BoundaryRuntime()
        sim.step()
        self.assertAlmostEqual(sim.physical_time,sim.config.dt)

    def test_queued_frame_is_validated_once_before_channel_access(self):
        sim = self.api().AstroCollisionSimulation()
        sim.runtime = Mock()
        sim.runtime.display_channels.return_value = {"positions": object()}
        sim.queue_viewport_step()
        with self.assertRaisesRegex(RuntimeError, "cannot queue"):
            sim.queue_viewport_step()
        self.assertEqual(sim.physical_time, 0)
        self.assertIs(sim.display_channels()["positions"], sim.runtime.display_channels.return_value["positions"])
        sim.finish_viewport_step()
        sim.runtime.finish_frame_substep.assert_called_once()
        sim.runtime.finish_frame.assert_called_once()
        self.assertEqual(sim.physical_time, sim.config.dt)

    def test_failed_queued_frame_hides_channels_and_drains_on_close(self):
        sim = self.api().AstroCollisionSimulation()
        runtime = sim.runtime = Mock()
        device = sim.device_context = Mock()
        runtime.finish_frame_substep.side_effect = RuntimeError("non-finite acceleration")
        sim.queue_viewport_step()
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            sim.display_channels()
        with self.assertRaisesRegex(RuntimeError, "reset"):
            sim.step()
        runtime.display_channels.assert_not_called()
        self.assertEqual(sim.physical_time, 0)
        sim.close()
        device.wp.synchronize_device.assert_called_once_with(device.device)
        runtime.close.assert_called_once()

    def test_compressible_refresh_reuses_gpu_radius_buffer_each_step(self):
        a=self.api();sim=a.AstroCollisionSimulation(a.SimulationConfig(mode='COMPRESSIBLE'))
        mass=object();rho=object();radius=object()
        class Writer:
            calls=0
            def update(self,channels):
                self.calls+=1
                self.assert_inputs=(channels['masses'] is mass,channels['densities'] is rho)
                return radius
        class Runtime:
            def display_channels(self):return {'masses':mass,'densities':rho}
        sim.runtime=Runtime();sim._radius_writer=Writer()
        sim._refresh_channels();sim._refresh_channels()
        self.assertIs(sim.display_channels()['radii'],radius)
        self.assertEqual(sim._radius_writer.calls,2)
        self.assertEqual(sim._radius_writer.assert_inputs,(True,True))

    def test_material_energy_labels_distinguish_iisph_heat_and_compressible_thermo(self):
        a=self.api()
        self.assertIn('proxy',a.SimulationConfig(mode='IMPACT').energy_label.lower())
        self.assertIn('specific',a.SimulationConfig(mode='COMPRESSIBLE').energy_label.lower())
        self.assertIn('collisionless',a.SimulationConfig(mode='GALAXY').energy_label.lower())


if __name__=='__main__':unittest.main()
