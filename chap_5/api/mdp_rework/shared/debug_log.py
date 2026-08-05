import time


def log_progress(current: int, total: int):
    now = time.perf_counter()

    if not hasattr(log_progress, "start_time"):
        log_progress.start_time = now
        log_progress.last_print_time = now

    if now - log_progress.last_print_time >= 10.0:
        elapsed = now - log_progress.start_time
        speed = current / elapsed if elapsed > 0 else 0
        remaining = (total - current) / speed if speed > 0 else 0

        print(f"  {current:,}/{total:,} | {speed:,.0f} seq/s | Temps restant: {remaining:,.0f}s")

        log_progress.last_print_time = now


def reset_progress():
    if hasattr(log_progress, "start_time"):
        del log_progress.start_time
        del log_progress.last_print_time
