"""Pruebas de interfaz: cada página de la app carga sin errores (AppTest)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES = ["Inicio.py", *sorted(p.relative_to(APP_DIR).as_posix()
                               for p in (APP_DIR / "pages").glob("*.py"))]


@pytest.fixture(autouse=True)
def _app_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(APP_DIR))
    yield
    sys.modules.pop("common", None)


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_page_loads_without_exceptions(page):
    at = AppTest.from_file(str(APP_DIR / page), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.slow
def test_home_loads_example_scenario():
    at = AppTest.from_file(str(APP_DIR / "Inicio.py"), default_timeout=120)
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state.cfg["network"]["n"] == 245
