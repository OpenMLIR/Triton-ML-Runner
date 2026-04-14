class CompiledTVMFFIKernel:
    def __init__(self, function, metadata):
        self._function = function
        self._metadata = metadata
        self._run_launcher = None

    @staticmethod
    def _normalize_args(args):
        import torch
        return tuple(arg.data_ptr() if isinstance(arg, torch.Tensor) else arg for arg in args)

    def _get_launcher(self):
        if self._run_launcher is None:
            from triton_runner.tvm_ffi.driver import TvmFfiLauncher
            self._run_launcher = TvmFfiLauncher(self._metadata, self._function)
        return self._run_launcher

    def launch(self, gridX, gridY, gridZ, *args):
        self._get_launcher().launch(gridX, gridY, gridZ, *args)

    def run(self, gridX, gridY, gridZ, launch_enter_hook, launch_exit_hook, *args):
        self.launch(gridX, gridY, gridZ, *args)

    def __call__(self, *args, grid=None):
        if grid is None:
            raise ValueError("grid must be provided when calling CompiledTVMFFIKernel directly")
        self.launch(grid[0], grid[1], grid[2], *self._normalize_args(args))

    def __getitem__(self, grid):
        launcher = self._get_launcher()

        def runner(*args, stream=None):
            launcher.launch(grid[0], grid[1], grid[2], *self._normalize_args(args))

        return runner

    @classmethod
    def from_cubin(cls, cubin_path, metadata=None):
        from pathlib import Path
        import json
        from triton.runtime import driver

        cubin_path = Path(cubin_path)
        cubin_bytes = cubin_path.read_bytes()
        if metadata is None:
            json_path = cubin_path.with_suffix(".json")
            if json_path.exists():
                metadata = json.loads(json_path.read_text())
            else:
                raise ValueError(f"metadata is None and no JSON file found at {json_path}")

        device = driver.active.get_current_device()
        module, function, n_regs, n_spills, n_max_threads = driver.active.utils.load_binary(
            metadata["name"], cubin_bytes, metadata.get("shared", 0), device
        )
        del module
        return cls(function, metadata)
