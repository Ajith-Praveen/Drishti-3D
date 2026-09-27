"""Tools > Preferences: switchable themes and typefaces."""

from __future__ import annotations

import pytest

from drishti3d.app import theme
from drishti3d.app.main_window import MainWindow
from drishti3d.app.panels.preferences import PreferencesDialog


@pytest.fixture(autouse=True)
def _restore_default_theme(qapp):
    yield
    theme.apply_theme(qapp, theme.DEFAULT_PALETTE, theme.DEFAULT_TYPEFACE)


def test_every_palette_defines_every_token():
    for palette in theme.PALETTES.values():
        assert set(palette.tokens) == set(theme._TOKEN_NAMES), palette.key


def test_confidence_colours_do_not_move_with_the_palette(qapp):
    before = (theme.CONF_MEASURED, theme.CONF_LOW, theme.CONF_INFERRED)
    for key in theme.PALETTES:
        theme.apply_theme(qapp, key, None)
        assert (theme.CONF_MEASURED, theme.CONF_LOW, theme.CONF_INFERRED) == before


def test_bundled_typefaces_resolve(qapp):
    theme.apply_theme(qapp, None, "plex")
    for face in theme.TYPEFACES.values():
        if face.ui_family is None:
            continue
        theme.set_typeface(face.key)
        assert theme.ui_font().family() == face.ui_family
        assert theme.mono_font().family() == face.mono_family


def test_apply_appearance_rebuilds_and_keeps_state(qtbot, qapp, monkeypatch):
    monkeypatch.setattr(theme, "save_preferences", lambda: None)  # don't touch real prefs
    window = MainWindow()
    qtbot.addWidget(window)
    window.settings_panel.window_spin.setValue(17)
    window._set_video("/tmp/does-not-exist.mp4")
    window.diagnostics.append_log("keep me")
    old_viewport = window.viewport

    window._apply_appearance("violet", "geist")

    assert window.viewport is not old_viewport
    assert theme.ACCENT == theme.PALETTES["violet"].tokens["ACCENT"]
    assert theme.ACCENT in qapp.styleSheet()
    assert window.settings_panel.window_spin.value() == 17
    assert window.video_path == "/tmp/does-not-exist.mp4"
    assert "keep me" in window.diagnostics.log_view.toPlainText()
    # Menus rebuilt once, not duplicated.
    titles = [a.text() for a in window.menuBar().actions()]
    assert titles.count("&Tools") == 1


def test_preferences_dialog_emits_selection(qtbot):
    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    received = []
    dialog.appearanceApplied.connect(lambda p, t: received.append((p, t)))
    for button in dialog.palette_group.buttons():
        if button.key == "carbon":
            button.setChecked(True)
    for button in dialog.type_group.buttons():
        if button.key == "barlow":
            button.setChecked(True)
    dialog._apply()
    assert received == [("carbon", "barlow")]
