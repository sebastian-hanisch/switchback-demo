"""Presets: Vollständigkeit, gültige Werte, Einrasten der Permalink-Werte."""

import swb_constants as C
import swb_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_on_the_grid():
    for preset in C.PRESETS.values():
        assert C.Q_MIN <= preset["q"] <= C.Q_MAX
        steps = preset["q"] / C.Q_STEP
        assert abs(steps - round(steps)) < 1e-9
        assert C.L_MIN <= preset["L"] <= C.L_MAX
        assert C.K_MIN <= preset["k"] <= C.K_MAX
        assert 0 <= preset["metric"] < len(C.METRICS)
        assert 0 <= preset["seed"] <= C.SEED_MAX


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_labels_are_complete():
    assert set(C.METHOD_LABELS) == set(C.METHOD_SHORT) == set(C.METHOD_COLORS) == set(C.METHODS)
    assert set(C.METRIC_LABELS) == set(C.METRICS)


def test_permalink_values_are_clamped_and_snapped():
    assert P.clamp_to_spec(P.SETTING_SPECS["q_slider"], 0.34) == 0.3
    assert P.clamp_to_spec(P.SETTING_SPECS["q_slider"], 7) == 1.0
    assert P.clamp_to_spec(P.SETTING_SPECS["l_slider"], 99) == 28
    assert P.clamp_to_spec(P.SETTING_SPECS["k_slider"], -4) == 0
    assert P.clamp_to_spec(P.SETTING_SPECS["metric_radio"], 5) == 1


def test_block_lengths_and_washout_are_compatible():
    assert C.DEFAULT_K >= C.LEAD
    assert C.DEFAULT_L > C.DEFAULT_K
