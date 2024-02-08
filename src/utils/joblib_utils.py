import contextlib
from typing import Optional
import joblib
from tqdm.auto import tqdm

"""
Usage:

with joblib_tqdm(100):
    results = Parallel(n_jobs=10)(
        compute_something(i) for i in range(100)
    )
"""

@contextlib.contextmanager
def joblib_tqdm(total: Optional[int] = None, **kwargs):
    
    pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        pbar.close()
