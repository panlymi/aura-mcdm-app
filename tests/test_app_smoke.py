from streamlit.testing.v1 import AppTest


def test_streamlit_app_initial_screen_loads_without_exception():
    app = AppTest.from_file("aura_app.py", default_timeout=15).run()
    assert not app.exception
    assert app.title[0].value.startswith("Multi-Criteria Decision Making")
    assert app.info[0].value.startswith("Please upload")
