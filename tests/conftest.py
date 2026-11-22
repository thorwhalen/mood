"""Keep mood's import-time app-data folders inside a temp dir."""

import os
import tempfile

os.environ.setdefault("MOOD_APP_DATA_DIR", tempfile.mkdtemp(prefix="mood_test_"))
os.environ.setdefault("SCOOPY_APP_DATA_DIR", tempfile.mkdtemp(prefix="scoopy_test_"))
