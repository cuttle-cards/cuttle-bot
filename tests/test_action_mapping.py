"""Sanity checks for fixed action mapping."""
from game.game import Game
from rl.action_mapping import (
    ACTION_SPACE_SIZE,
    action_index_to_action,
    action_to_index,
    build_action_map,
    legal_action_mask_from_actions,
)


def test_action_mask_roundtrip() -> None:
    """Ensure mask/index mapping round-trips for legal actions in a fresh game state."""
    game = Game(manual_selection=False, ai_player=None)
    state = game.game_state
    legal_actions = state.get_legal_actions()

    action_map = build_action_map(legal_actions)
    mask = legal_action_mask_from_actions(legal_actions)

    assert mask.shape[0] == ACTION_SPACE_SIZE
    assert int(mask.sum()) == len(action_map)

    for idx, action in action_map.items():
        assert mask[idx]
        decoded = action_index_to_action(state, idx)
        assert decoded is not None
        assert action_to_index(decoded) == idx

    for action in legal_actions:
        idx = action_to_index(action)
        assert idx is not None
        assert mask[idx]
