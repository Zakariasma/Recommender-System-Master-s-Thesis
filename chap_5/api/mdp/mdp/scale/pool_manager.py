import time
import multiprocessing as mp


def chunkify(n_states, n_chunks):
    size = max(1, n_states // n_chunks)
    return [(i, min(i + size, n_states)) for i in range(0, n_states, size)]


def run_parallel_tasks(worker_fn, tasks, total_states, n_workers, initializer, initargs):
    counter = mp.Value("l", 0)
    start_time = time.time()

    with mp.Pool(n_workers, initializer=initializer, initargs=initargs + (counter,)) as pool:
        results = pool.map_async(worker_fn, tasks)
        while not results.ready():
            time.sleep(1.0)
            elapsed = time.time() - start_time
            done = counter.value
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total_states - done) / speed / 60 if speed > 0 else 0
            print(f"\r  Progression: {done:,}/{total_states:,} | {speed:,.0f} états/s | ETA: {eta:.1f} min", end="",
                  flush=True)
        print()
    return results.get()