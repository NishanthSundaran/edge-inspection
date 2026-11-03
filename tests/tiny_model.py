"""A tiny OpenVINO model for service tests and the CI container smoke test: score = mean brightness.

  python -m tests.tiny_model deploy/tiny
"""

import json
import pathlib
import sys

import numpy as np

S = 32


def write_tiny_model(d) -> None:
    import openvino as ov
    import openvino.opset13 as ops
    d = pathlib.Path(d)
    d.mkdir(parents=True, exist_ok=True)
    x = ops.parameter([1, 3, S, S], np.float32, name="image")
    amap = ops.reduce_mean(x, np.array([1], np.int64), keep_dims=False)          # (1, S, S)
    score = ops.reduce_mean(amap, np.array([1, 2], np.int64), keep_dims=False)    # (1,)
    ov.save_model(ov.Model([score, amap], [x], "tiny"), str(d / "model.xml"))
    (d / "meta.json").write_text(json.dumps({"category": "tiny", "size": S, "threshold": 0.5,
                                             "calib_median": 0.2, "calib_std": 0.05, "runtime": "openvino-test"}))


if __name__ == "__main__":
    write_tiny_model(sys.argv[1])
