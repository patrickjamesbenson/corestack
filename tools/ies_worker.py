
from __future__ import annotations
import types

def _streamlit_stub() -> types.ModuleType:
    # Minimal stub used by legacy modules so imports don't crash in headless mode
    def _noop(*a, **k): return None
    def _false(*a, **k): return False
    st = types.SimpleNamespace(
        set_page_config=_noop, button=_false, checkbox=_false, selectbox=_noop, slider=_noop,
        text_input=lambda *a, **k: "", number_input=lambda *a, **k: 0,
        file_uploader=_noop, write=_noop, markdown=_noop, code=_noop, caption=_noop,
        success=_noop, info=_noop, warning=_noop, error=_noop, subheader=_noop, header=_noop,
        columns=lambda *a, **k: [], experimental_rerun=_noop,
    )
    return st
