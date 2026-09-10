import ast
import json
import os
from pathlib import Path

from ..compiler.source_types import DUMP_IR_DIR_TYPES, METADATA_DIR_TYPES, METADATA_INLINE_SRC_TYPES


class MetadataMixin:
    def handle_autotune(self, kwargs):
        if kwargs.get("autotune_cubin_dir"):
            metadata_json = self.get_metadata_json(kwargs["autotune_cubin_dir"])
            kernel_signature_tuple = ast.literal_eval(metadata_json["kernel_signature"])
            for (key, arg_type, spec, is_kwargs) in kernel_signature_tuple:
                if is_kwargs:
                    kwargs[key] = spec
            kwargs["cubin_dir"] = kwargs["autotune_cubin_dir"]
            kwargs["num_warps"] = metadata_json["num_warps"]
            kwargs["num_stages"] = metadata_json["num_stages"]
            kwargs["num_ctas"] = metadata_json["num_ctas"]
            kwargs["maxnreg"] = metadata_json["maxnreg"]
            # pre_hook and ir_override is disabled

    def get_metadata_json(self, path):
        json_file_name = f"{self.__name__}.json"
        json_path = os.path.join(path, json_file_name)
        cubin_file_name = f"{self.__name__}.cubin"
        cubin_path = os.path.join(path, cubin_file_name)
        if not (os.path.exists(json_path) and os.path.exists(cubin_path)):
            from triton.runtime.errors import PTXASError
            raise PTXASError("autotune_cubin_dir is error")
        return json.loads(Path(json_path).read_text())

    def get_src_and_metadata_json(self, kwargs, source_dir_type, src, ast_src):
        if source_dir_type:
            if source_dir_type.endswith("src"):
                value = kwargs[source_dir_type]
                src_ext = source_dir_type[:-4]  # "ttgir" from "ttgir_src"
                content = Path(value).read_text() if os.path.exists(value) else value
                runner_cache_dir = self.get_runner_cache_dir()
                src = os.path.join(runner_cache_dir, f"{self.__name__}-src.{src_ext}")
                Path(src).write_text(content)
            else:
                source_file_name = f"{self.__name__}.{source_dir_type[:-4]}"
                src = os.path.join(kwargs[source_dir_type], source_file_name)

            if self.need_dump(kwargs) and source_dir_type in DUMP_IR_DIR_TYPES:
                if not os.path.exists(src):
                    src = os.path.join(kwargs[source_dir_type], source_file_name[:-4] + "source")
                if not os.path.exists(src):
                    raise RuntimeError("Check .source/.ttir/.ttgir file for dump.")
                dump_content = self.insert_dump_tensor_param(Path(src).read_text())
                dump_content = self.inject_ssa_ir_dump_store(dump_content, kwargs["dump_value"], kwargs.get("dump_grid", 0))
                src = os.path.join(kwargs[source_dir_type], f"dump.{source_dir_type[:-4]}")
                Path(src).write_text(dump_content)
            metadata_json = {}
            if source_dir_type in METADATA_DIR_TYPES:
                json_file_name = f"{self.__name__}.json"
                json_path = os.path.join(kwargs[source_dir_type], json_file_name)
                metadata_json = json.loads(Path(json_path).read_text())
            elif source_dir_type == "ttgir_dir":
                json_path = os.path.join(kwargs[source_dir_type], f"{self.__name__}.json")
                if os.path.exists(json_path):
                    metadata_json = json.loads(Path(json_path).read_text())
            elif source_dir_type in METADATA_INLINE_SRC_TYPES:
                metadata_json = kwargs.get("metadata_json") or {}
                if not metadata_json:
                    raise ValueError(f"{source_dir_type} requires metadata_json")
            return src, metadata_json
        elif self.need_dump(kwargs):
            return src, {}
        return ast_src, {}
