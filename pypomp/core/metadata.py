import importlib.metadata
import platform
import datetime
import jax
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ModelMetadata:
    """Stores environment and instantiation metadata for reproducibility."""
    
    pypomp_version: str = field(
        default_factory=lambda: importlib.metadata.version("pypomp")
    )
    jax_version: str = field(default_factory=lambda: jax.__version__)
    python_version: str = field(default_factory=platform.python_version)
    platform_info: str = field(default_factory=platform.platform)
    default_device: str = field(default_factory=lambda: str(jax.default_backend()))
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def print_metadata(self) -> None:
        """Prints all the metadata fields cleanly to the console."""
        print("Model Initialization Metadata:")
        print("------------------------------")
        print(f"pypomp version: {self.pypomp_version}")
        print(f"JAX version:    {self.jax_version}")
        print(f"Python version: {self.python_version}")
        print(f"Platform info:  {self.platform_info}")
        print(f"Default device: {self.default_device}")
        print(f"Created at:     {self.created_at}")
