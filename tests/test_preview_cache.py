"""Full-count local geometry reuse, invalidation, and bounded retention."""
from dataclasses import replace
import importlib
import unittest
from unittest.mock import patch

import numpy as np
from astro_collision_lab.simulation import BodySpec, SimulationConfig

try:
    PreviewCache = importlib.import_module('astro_collision_lab.preview_cache').PreviewCache
except ModuleNotFoundError:
    PreviewCache = None


class PreviewCacheTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(PreviewCache, 'local preview cache is not implemented')
        self.cache = PreviewCache()
        self.body = BodySpec('Disk', kind='DISK', particle_count=257, radius=3.,
                             scale_length=.6, thickness=.2, seed=9)
        self.config = SimulationConfig(mode='GALAXY', particle_count=257, seed=4)

    def test_rigid_transform_name_membership_and_motion_reuse_exact_array(self):
        key, points = self.cache.sample(self.body, self.config)
        moved = replace(self.body, name='Renamed', galaxy_id='other', position=(4.,2.,1.),
                        rotation=((0.,-1.,0.),(1.,0.,0.),(0.,0.,1.)), mass=5.,
                        velocity=(4.,0.,0.), spin=(0.,0.,3.), toomre_q=2., rotation_fraction=-1.)
        moved_key, moved_points = self.cache.sample(moved, self.config)
        self.assertEqual(moved_key, key)
        self.assertIs(moved_points, points)
        self.assertEqual(points.shape, (257,3))
        self.assertTrue(np.isfinite(points).all())
        np.testing.assert_allclose(points.mean(0), 0., atol=1e-12)

    def test_first_sample_is_local_even_when_source_is_transformed(self):
        moved = replace(self.body, position=(100.,20.,-3.),
                        rotation=((0.,-1.,0.),(1.,0.,0.),(0.,0.,1.)))
        key, points = self.cache.sample(moved, self.config)
        fresh_key, local = PreviewCache().sample(self.body, self.config)
        self.assertEqual(key, fresh_key)
        np.testing.assert_array_equal(points, local)

    def test_geometry_and_sampling_edits_regenerate_full_count(self):
        key, points = self.cache.sample(self.body, self.config)
        for field, value in [('radius',4.),('seed',10),('particle_count',500),
                             ('scale_length',.7),('thickness',.3),('kind','HALO')]:
            with self.subTest(field=field):
                changed=replace(self.body, **{field:value})
                new_key, new_points=self.cache.sample(changed,self.config)
                self.assertNotEqual(new_key,key)
                self.assertIsNot(new_points,points)
                self.assertEqual(new_points.shape,(changed.particle_count,3))
        for config in (replace(self.config,seed=8),replace(self.config,mode='IMPACT')):
            self.assertNotEqual(self.cache.sample(self.body,config)[0], key)
        self.assertEqual(self.cache.sample(self.body,replace(self.config,dt=.02,theta=.4))[0],key)

    def test_lru_eviction_retains_recently_used_entry_and_clear_releases(self):
        cache=PreviewCache(limit=2)
        first=cache.sample(self.body,self.config)
        second=cache.sample(replace(self.body,seed=10),self.config)
        self.assertIs(cache.sample(self.body,self.config)[1],first[1])
        cache.sample(replace(self.body,seed=11),self.config)
        self.assertIs(cache.sample(self.body,self.config)[1],first[1])
        self.assertIsNot(cache.sample(replace(self.body,seed=10),self.config)[1],second[1])
        cache.clear()
        self.assertIsNot(cache.sample(self.body,self.config)[1],first[1])

    def test_invalid_generator_output_is_rejected_and_never_cached(self):
        for invalid in (np.zeros((256,3)), np.full((257,3),np.nan)):
            with self.subTest(shape=invalid.shape), patch('astro_collision_lab.galaxy_models.preview',return_value=[invalid]):
                with self.assertRaises(ValueError):
                    self.cache.sample(self.body,self.config)
        self.assertEqual(self.cache.sample(self.body,self.config)[1].shape,(257,3))

    def test_invalid_budget_and_nonfinite_geometry_are_rejected(self):
        for count in (0,-1,True,1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.cache.sample(replace(self.body,particle_count=count),self.config)
        with self.assertRaises(ValueError):
            self.cache.sample(replace(self.body,radius=float('nan')),self.config)
        for limit in (0,-1,True,1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                PreviewCache(limit=limit)


if __name__=='__main__':
    unittest.main()
