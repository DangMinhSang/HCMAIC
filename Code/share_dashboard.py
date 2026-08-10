"""Expose the custom Flask dashboard through a Gradio share link on Kaggle."""

from __future__ import annotations

import gradio as gr
from starlette.middleware.wsgi import WSGIMiddleware

from dashboard import app as dashboard_app


def launch_dashboard(share: bool = True) -> str | None:
    """Launch a small Gradio gateway and mount the full dashboard at /dashboard.

    Kaggle notebooks require Gradio sharing for a browser-accessible URL. The
    gateway only provides the tunnel; all UI/API requests are handled by the
    custom Flask dashboard and use the same mounted Kaggle inputs.
    """
    with gr.Blocks(title="AIC26 Dashboard") as gateway:
        gr.Markdown("# AIC26 Dashboard\n\n[Open the retrieval dashboard →](./dashboard/)")
    # Blocks creates its FastAPI app during launch, so mount the WSGI app
    # immediately after launch returns while the gateway server is alive.
    gateway_app, _local_url, share_url = gateway.launch(
        share=share,
        prevent_thread_lock=True,
    )
    gateway_app.mount("/dashboard", WSGIMiddleware(dashboard_app))
    if share_url:
        return f"{share_url.rstrip('/')}/dashboard/"
    return None
