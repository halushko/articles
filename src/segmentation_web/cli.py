from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("segmentation_web.main:app", host="0.0.0.0", port=8080)
