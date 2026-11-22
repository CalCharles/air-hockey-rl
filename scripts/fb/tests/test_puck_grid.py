import numpy as np
import torch
from scripts.fb.evaluate_puck_grid import goal_weights, post_contact_distances, grid_goals


def test_contacts_are_required_and_latched():
    positions = [[0, 0], [1, 0], [2, 0]]
    goals = np.array([[0, 0], [2, 0]])
    np.testing.assert_allclose(post_contact_distances(positions, [False, True, False], goals), [1, 0])
    assert np.isinf(post_contact_distances(positions, [False]*3, goals)).all()


def test_inference_ignores_occlusion_placeholders():
    obs = torch.zeros(3, 32)
    obs[1, 29] = 1
    obs[2, 27] = 1
    weights, support = goal_weights(obs, [0, 0], .05)
    assert support == 1
    assert weights[0] == 1
    assert weights[1] == 0
    assert weights[2] < 1e-10


def test_grid_is_upper_half_and_inset():
    class Env:
        puck_radius = .03175
        table_x_top = -.9652
        table_y_left = -.4318
        table_y_right = .4318
    goals = grid_goals(Env(), 5, .05)
    assert goals.shape == (25, 2)
    assert np.all(goals[:, 0] < 0)
    assert np.all(goals[:, 0] >= Env.table_x_top + .08174)
    assert np.max(np.abs(goals[:, 1])) <= .35006


def test_scoring_reads_physics_not_public_observed_state():
    from types import SimpleNamespace
    from scripts.fb.evaluate_puck_grid import true_puck_position
    env = SimpleNamespace(
        current_state={"pucks": [{"position": [-.8, 0]}]},
        simulator=SimpleNamespace(
            pucks={"puck_0": SimpleNamespace(position=[.2, .4])},
            _box2d_to_base_coords=lambda pos: [-pos[1], pos[0]]))
    np.testing.assert_allclose(true_puck_position(env), [-.4, .2])
