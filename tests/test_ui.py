"""Streamlit UI Integration Tests for TracePulse Dashboard.

Uses streamlit.testing.v1.AppTest to execute the application headless,
test user interactions (sliders, toggles, regeneration), and verify KPI metrics
and incident cards without opening a browser.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class TestStreamlitApp:
    """Verifies that app.py boots, renders KPIs, and handles widget updates without exceptions."""

    def test_dashboard_initial_run(self):
        """Verify the app launches with default sample logs and renders KPIs."""
        at = AppTest.from_file(APP_PATH).run(timeout=30)
        assert not at.exception, f"App threw unexpected exception: {at.exception}"

        # Verify metrics (Total Logs Ingested, Anomalies Flagged, Grouped Incidents, ML Inference Time)
        assert len(at.metric) >= 4
        labels = [m.label for m in at.metric]
        assert "Total Logs Ingested" in labels
        assert "Anomalies Flagged" in labels
        assert "Grouped Incidents" in labels
        assert "ML Inference Time" in labels

        # Total logs should be populated (e.g. ~1500+)
        total_logs_val = at.metric[0].value
        assert total_logs_val and total_logs_val != "0"

    def test_tune_contamination_slider(self):
        """Verify adjusting the contamination slider triggers recalculation without errors."""
        at = AppTest.from_file(APP_PATH).run(timeout=30)
        assert not at.exception

        # Find contamination slider (first slider)
        if at.slider:
            contamination_slider = at.slider[0]
            contamination_slider.set_value(0.05).run(timeout=30)
            assert not at.exception
            assert len(at.metric) >= 4

    def test_toggle_heuristic_and_llm_mode(self):
        """Verify toggling between Heuristic and Live LLM mode does not crash."""
        at = AppTest.from_file(APP_PATH).run(timeout=30)
        assert not at.exception

        if at.toggle:
            llm_toggle = at.toggle[0]
            current_val = llm_toggle.value
            llm_toggle.set_value(not current_val).run(timeout=30)
            assert not at.exception
            llm_toggle.set_value(current_val).run(timeout=30)
            assert not at.exception
