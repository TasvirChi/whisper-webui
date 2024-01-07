from typing import List
from src.alignment.alignment import Alignment
from src.diarization.diarization import Diarization, DiarizationEntry
from src.modelCache import GLOBAL_MODEL_CACHE, ModelCache
from src.vadParallel import ParallelContext

class AlignmentContainer:
    def __init__(self, enable_daemon_process: bool = True, auto_cleanup_timeout_seconds=60, cache: ModelCache = None):
        self.enable_daemon_process = enable_daemon_process
        self.auto_cleanup_timeout_seconds = auto_cleanup_timeout_seconds
        self.alignment_context: ParallelContext = None
        self.cache = cache
        self.model = None

    def run(self, audio_file, result, **kwargs):
        # Create parallel context if needed
        if self.alignment_context is None and self.enable_daemon_process:
            # Number of processes is set to 1 as we mainly use this in order to clean up GPU memory
            self.alignment_context = ParallelContext(num_processes=1, auto_cleanup_timeout_seconds=self.auto_cleanup_timeout_seconds)
            print("Created alignment context with auto cleanup timeout of %d seconds" % self.auto_cleanup_timeout_seconds)
        
        # Run directly 
        if self.alignment_context is None:
            return self.execute(audio_file, result, **kwargs)

        # Otherwise run in a separate process
        pool = self.alignment_context.get_pool()

        try:
            result = pool.apply(self.execute, (audio_file,), kwargs)
            return result
        finally:
            self.alignment_context.return_pool(pool)

    def get_model(self):
        # Lazy load the model
        if (self.model is None):
            if self.cache:
                print("Loading alignment model from cache")
                self.model = self.cache.get("alignment", lambda : Alignment())
            else:
                print("Loading diarization model")
                self.model = Alignment()
        return self.model

    def execute(self, audio_file, result, **kwargs):
        model = self.get_model()

        # We must use list() here to force the iterator to run, as generators are not picklable
        result = list(model.run(audio_file, **kwargs))
        return result
    
    def cleanup(self):
        if self.alignment_context is not None:
            self.alignment_context.close()

    def __getstate__(self):
        return {
            "auth_token": self.auth_token,
            "enable_daemon_process": self.enable_daemon_process,
            "auto_cleanup_timeout_seconds": self.auto_cleanup_timeout_seconds
        }
    
    def __setstate__(self, state):
        self.auth_token = state["auth_token"]
        self.enable_daemon_process = state["enable_daemon_process"]
        self.auto_cleanup_timeout_seconds = state["auto_cleanup_timeout_seconds"]
        self.alignment_context = None
        self.cache = GLOBAL_MODEL_CACHE
        self.model = None