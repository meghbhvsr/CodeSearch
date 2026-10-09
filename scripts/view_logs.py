"""Launch TensorBoard with TensorFlow hidden.

The installed TensorFlow is broken ("module 'tensorflow' has no attribute 'io'"). Plain
`tensorboard` still starts, but it fails to read any logs and shows empty charts.
Hiding TensorFlow makes it use its built-in TensorFlow-free reader.

  python scripts/view_logs.py                     # all runs under checkpoints/
  python scripts/view_logs.py --logdir checkpoints/mnrl
"""

import sys
from pathlib import Path

sys.modules["tensorflow"] = None

from tensorboard.main import run_main  # noqa: E402

if "--logdir" not in sys.argv:
    sys.argv += ["--logdir", str(Path(__file__).resolve().parent.parent / "checkpoints")]
run_main()
