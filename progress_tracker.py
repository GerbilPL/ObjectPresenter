import threading


class ProgressTracker:
    """
    Thread-safe tracker for progress and cancellation.
    Designed to be used both in GUI and headless CLI modes.
    """

    def __init__(self) -> None:
        """Initializes thread-safe progress tracking with default state."""
        self._lock = threading.Lock()
        self.total_items = 1
        self.current_item = 0
        self.item_progress = 0.0  # Range: 0.0 to 100.0
        self.status = ""
        self.is_cancelled = False

    def start_batch(self, total_items: int) -> None:
        """Initializes progress tracker for a new batch of tasks.
        
        Args:
            total_items: Total number of items to process in batch.
                Must be >= 1, automatically clamped to 1 if smaller.
        """
        with self._lock:
            self.total_items = max(1, total_items)
            self.current_item = 0
            self.item_progress = 0.0
            self.is_cancelled = False
            self.status = "Starting batch processing..."

    def set_current_item(self, index: int, status: str = "") -> None:
        """Updates tracker to a specific item and resets item-level progress.
        
        Args:
            index: Zero-based index of current item in batch.
            status: Optional status message (empty string = no update).
        """
        with self._lock:
            self.current_item = index
            self.item_progress = 0.0
            if status:
                self.status = status

    def update_progress(self, progress: float, status: str = None) -> None:
        """Updates item-level progress and optionally status message.
        
        Args:
            progress: Item progress 0.0-100.0. Automatically clamped to valid range.
            status: Optional status message (None = no update, existing status kept).
        """
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
        """Checks if cancellation was requested by user.
        
        Raises:
            InterruptedError: If is_cancelled flag is True. Worker threads should
                catch and handle this to gracefully stop processing.
        """
        with self._lock:
            if self.is_cancelled:
                raise InterruptedError("Process was cancelled by the user.")

    def get_state(self) -> dict:
        """Returns thread-safe snapshot of current progress state.
        
        Returns:
            Dict with keys:
                'overall_progress': 0.0-100.0, batch-wide progress.
                'item_progress': 0.0-100.0, current item progress.
                'status': Status message string.
                'is_cancelled': Boolean cancellation flag.
        """
        with self._lock:
            # Calculate overall progress across the entire batch
            overall = ((self.current_item + (self.item_progress / 100.0)) / self.total_items) * 100.0

            return {
                "overall_progress": min(100.0, overall),
                "item_progress": self.item_progress,
                "status": self.status,
                "is_cancelled": self.is_cancelled
            }