import sys
import time
from typing import Optional, Callable, Any
from threading import Thread, Event
from contextlib import contextmanager

class Spinner:
    """A simple terminal spinner for indicating progress."""
    
    def __init__(
        self,
        message: str = "Loading...",
        delay: float = 0.1,
        spinner_chars: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    ):
        self.message = message
        self.delay = delay
        self.spinner_chars = spinner_chars
        self.stop_event = Event()
        self.spinner_thread: Optional[Thread] = None

    def _spin(self):
        """The actual spinner animation."""
        i = 0
        while not self.stop_event.is_set():
            sys.stdout.write(f"\r{self.spinner_chars[i]} {self.message}")
            sys.stdout.flush()
            i = (i + 1) % len(self.spinner_chars)
            time.sleep(self.delay)

    def start(self):
        """Start the spinner animation."""
        self.stop_event.clear()
        self.spinner_thread = Thread(target=self._spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()

    def stop(self, final_message: Optional[str] = None):
        """Stop the spinner and optionally display a final message."""
        if self.spinner_thread is None:
            return

        self.stop_event.set()
        self.spinner_thread.join()
        
        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
        
        if final_message:
            print(final_message)
        sys.stdout.flush()

    def update_message(self, message: str):
        """Update the spinner message."""
        self.message = message

class ProgressTracker:
    """A simple progress tracker that shows percentage completion."""
    
    def __init__(
        self,
        total: int,
        message: str = "Progress",
        width: int = 40
    ):
        self.total = total
        self.current = 0
        self.message = message
        self.width = width
        self.start_time = time.time()

    def update(self, increment: int = 1):
        """Update the progress."""
        self.current += increment
        self._display()

    def _display(self):
        """Display the current progress."""
        percentage = (self.current / self.total) * 100
        filled = int(self.width * self.current / self.total)
        bar = "█" * filled + "░" * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            rate = elapsed / self.current
            eta = rate * (self.total - self.current)
            time_str = f"ETA: {eta:.1f}s"
        else:
            time_str = "Calculating..."

        sys.stdout.write(f"\r{self.message}: [{bar}] {percentage:.1f}% {time_str}")
        sys.stdout.flush()

    def finish(self, final_message: Optional[str] = None):
        """Complete the progress bar and optionally show a final message."""
        self._display()
        print()  # New line
        if final_message:
            print(final_message)

@contextmanager
def progress_context(message: str = "Loading...", success_message: str | None = None, error_message: str | None = None):
    """Context manager for showing a spinner during an operation."""
    spinner = Spinner(message)
    spinner.start()
    try:
        yield spinner
        spinner.stop(success_message)
    except Exception as e:
        spinner.stop(error_message)
        raise

def with_progress(message: str = "Loading..."):
    """Decorator to add a spinner to a function."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            with progress_context(message) as spinner:
                return func(*args, **kwargs)
        return wrapper
    return decorator