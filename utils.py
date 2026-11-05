import inspect
import io
import random
import sys
import traceback
from collections.abc import Callable
from contextlib import redirect_stdout

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def train_with_logging(args, train_fn: Callable, writer: SummaryWriter | None = None):
    log_stream = io.StringIO()

    class TeeOutput(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
            return len(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    tee = TeeOutput(sys.stdout, log_stream)

    if writer:
        caller_frame = inspect.stack()[1]
        caller_filename = caller_frame.filename
        with open(caller_filename, "r") as f:
            source_code = f.read()
        with open("intrinsic_reward_module.py", "r") as f:
            irm_code = f.read()
        print("Logging source code to TensorBoard", flush=True)
        writer.add_text(
            "source_code",
            f"ppo_rin.py:\n```\n{source_code}\n```\n\nintrinsic_reward_module.py:\n```\n{irm_code}\n```",
        )
        writer.flush()

    try:
        with redirect_stdout(tee):
            train_fn(args, writer=writer)
    except BaseException as e:
        if isinstance(e, SystemExit):
            raise
        tb = traceback.format_exc()
        print("An exception occurred during training:")
        print(tb)
        log_stream.write(tb)
        if writer:
            print("Logging exception to TensorBoard", flush=True)
            writer.add_text("training_logs", f"Exception Traceback:\n```\n{tb}\n```")
            writer.add_text("training_logs", f"```\n{log_stream.getvalue()}\n```")
            writer.flush()
            writer.close()
        raise
    else:
        if writer:
            writer.add_text("training_logs", f"```\n{log_stream.getvalue()}\n```")
            writer.flush()
            writer.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
