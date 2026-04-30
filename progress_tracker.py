import threading


class ProgressTracker:
    """
    Thread-safe tracker for progress and cancellation.
    Designed to be used both in GUI and headless CLI modes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_items = 1
        self.current_item = 0
        self.item_progress = 0.0  # Range: 0.0 to 100.0
        self.status = ""
        self.is_cancelled = False

    def start_batch(self, total_items: int) -> None:
        """Initializes state for a new batch of tasks."""
        with self._lock:
            self.total_items = max(1, total_items)
            self.current_item = 0
            self.item_progress = 0.0
            self.is_cancelled = False
            self.status = "Starting batch processing..."

    def set_current_item(self, index: int, status: str = "") -> None:
        """Moves the tracker to the next item in the batch."""
        with self._lock:
            self.current_item = index
            self.item_progress = 0.0
            if status:
                self.status = status

    def update_progress(self, progress: float, status: str = None) -> None:
        """Updates progress of the current item."""
        with self._lock:
            self.item_progress = max(0.0, min(100.0, progress))
            if status is not None:
                self.status = status

    def cancel(self) -> None:
        """Triggers the cancellation flag."""
        with self._lock:
            self.is_cancelled = True
            self.status = "Canceling... please wait."

    def check_cancelled(self) -> None:
        """Raises InterruptedError if cancellation was requested."""
        with self._lock:
            if self.is_cancelled:
                raise InterruptedError("Process was cancelled by the user.")

    def get_state(self) -> dict:
        """Returns a snapshot of the current progress (thread-safe)."""
        with self._lock:
            # Calculate overall progress across the entire batch
            overall = ((self.current_item + (self.item_progress / 100.0)) / self.total_items) * 100.0

            return {
                "overall_progress": min(100.0, overall),
                "item_progress": self.item_progress,
                "status": self.status,
                "is_cancelled": self.is_cancelled
            }