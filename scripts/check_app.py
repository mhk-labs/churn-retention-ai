"""Smoke-test the Streamlit app without running a web server."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
app = AppTest.from_file(str(app_path)).run()
assert not app.exception, app.exception
assert app.metric[0].value == "72%"
app.selectbox[0].select("demo-002").run()
assert not app.exception, app.exception
assert app.metric[0].value == "18%"
print("Streamlit smoke test passed")
