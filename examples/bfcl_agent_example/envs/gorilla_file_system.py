"""Implementation of GorillaFileSystem for BFCL evaluation."""

from typing import Dict, Optional, Union


class File:
    """A file in the Gorilla File System."""

    def __init__(
        self, name: str = "", content: str = "", parent: Optional["Directory"] = None
    ):
        self.name: str = name
        self.content: str = content
        self.parent: Optional["Directory"] = parent

    def __repr__(self):
        return f"<File: {self.name}, Content: '{self.content[:20]}{'...' if len(self.content) > 20 else ''}'>"

    def __eq__(self, other):
        if not isinstance(other, File):
            return False
        return self.name == other.name and self.content == other.content


class Directory:
    """A directory in the Gorilla File System."""

    def __init__(
        self,
        name: str = "",
        parent: Optional["Directory"] = None,
        contents: Optional[Dict[str, Union[File, "Directory"]]] = None,
    ):
        self.name: str = name
        self.parent: Optional["Directory"] = parent
        self.contents: Dict[str, Union[File, "Directory"]] = contents or {}

    def __repr__(self):
        parent_name = self.parent.name if self.parent and self.parent.name else None
        return f"<Directory: {self.name}, Parent: {parent_name}, Keys: {list(self.contents.keys())}>"

    def __eq__(self, other):
        if not isinstance(other, Directory):
            return False
        return self.name == other.name and self.contents == other.contents


class GorillaFileSystem:
    """A file system for BFCL evaluation."""

    def __init__(self):
        self.root: Directory = Directory(name="workspace", parent=None)
        self.current_dir: Directory = self.root
        self.long_context: bool = False

    def _load_scenario(self, config: Dict):
        """Load the file system from configuration."""
        if not config or "root" not in config:
            # self.root and self.current_dir are already initialized with defaults
            pass
        elif "root" in config:
            try:
                loaded_dir: Optional[Directory] = None
                root_config = config["root"]
                if isinstance(root_config, dict) and "type" in root_config:
                    loaded_dir = self._load_directory_from_config(
                        "workspace", None, root_config
                    )
                elif isinstance(root_config, dict):
                    loaded_dir = self._load_directory_from_yaml_config(
                        "workspace", None, root_config
                    )

                if loaded_dir:
                    self.root = loaded_dir
                    self.current_dir = self.root
            except Exception as e:
                print(f"Error loading GorillaFileSystem scenario: {e}")
                self.root = Directory(name="workspace", parent=None)
                self.current_dir = self.root

        if "long_context" in config:  # Ensure config is checked before access
            self.long_context = config.get("long_context", False)

    def _load_directory_from_config(
        self, name: str, parent: Optional["Directory"], config: Dict
    ) -> Optional[Directory]:
        """Create a directory structure from configuration."""
        if config.get("type") == "directory":
            directory = Directory(name=name, parent=parent)
            contents: Dict[str, Union[File, Directory]] = {}
            for item_name, item_config in config.get("contents", {}).items():
                item_type = item_config.get("type")
                if item_type == "directory":
                    loaded_item = self._load_directory_from_config(
                        item_name, directory, item_config
                    )
                    if loaded_item:
                        contents[item_name] = loaded_item
                elif item_type == "file":
                    contents[item_name] = File(
                        name=item_name,
                        content=item_config.get("content", ""),
                        parent=directory,
                    )
            directory.contents = contents
            return directory
        return None

    def _load_directory_from_yaml_config(
        self, name: str, parent: Optional["Directory"], config: Dict
    ) -> Directory:
        """Create a directory structure from YAML configuration format."""
        directory = Directory(name=name, parent=parent)
        contents: Dict[str, Union[File, "Directory"]] = {}

        config_contents = config.get("contents", {})
        if not isinstance(config_contents, dict):  # Ensure it's a dict
            config_contents = {}

        for item_name, item_config in config_contents.items():
            if isinstance(item_config, dict):
                if "contents" in item_config:
                    loaded_subdir = self._load_directory_from_yaml_config(
                        item_name, directory, item_config
                    )
                    if loaded_subdir:  # Should always be true as per current impl
                        contents[item_name] = loaded_subdir
                elif "content" in item_config:
                    contents[item_name] = File(
                        name=item_name,
                        content=item_config.get("content", ""),
                        parent=directory,
                    )
                elif item_config.get("type") == "directory":
                    loaded_subdir = self._load_directory_from_yaml_config(
                        item_name, directory, item_config
                    )
                    if loaded_subdir:
                        contents[item_name] = loaded_subdir
                elif item_config.get("type") == "file":
                    contents[item_name] = File(
                        name=item_name,
                        content=item_config.get("content", ""),
                        parent=directory,
                    )
        directory.contents = contents
        return directory

    def pwd(self) -> Dict:
        """Return the current working directory path."""
        # self.current_dir and self.root are guaranteed to be Directory instances
        path_parts = []
        temp_dir: Optional[Directory] = self.current_dir

        while temp_dir is not None and temp_dir != self.root:
            path_parts.append(temp_dir.name)
            temp_dir = temp_dir.parent  # temp_dir.parent can be None

        if self.root:  # self.root is always a Directory
            path_parts.append(self.root.name)
        else:  # Should not happen given __init__
            return {"error": "Root directory not found unexpectedly."}

        full_path = "/" + "/".join(reversed(path_parts)) if path_parts else "/"
        return {"current_working_directory": full_path}

    def touch(self, file_name: str) -> Dict:
        """Create a new empty file or update timestamp (noop for content)."""
        # self.current_dir is Directory
        if file_name in self.current_dir.contents:
            item = self.current_dir.contents[file_name]
            if isinstance(item, Directory):
                return {"error": f"Cannot touch '{file_name}': It is a directory."}
            return {}

        self.current_dir.contents[file_name] = File(
            name=file_name, content="", parent=self.current_dir
        )
        return {}

    def rm(self, file_name: str) -> Dict:
        """Remove a file or directory."""
        # self.current_dir is Directory
        if file_name not in self.current_dir.contents:
            return {"result": f"Error: '{file_name}' not found."}

        del self.current_dir.contents[file_name]
        return {"result": f"Successfully removed '{file_name}'."}

    def rmdir(self, dir_name: str) -> Dict:
        """Remove an empty directory."""
        # self.current_dir is Directory
        if dir_name not in self.current_dir.contents:
            return {"result": f"Error: Directory '{dir_name}' not found."}

        item_to_remove = self.current_dir.contents[dir_name]

        if not isinstance(item_to_remove, Directory):
            return {"result": f"Error: '{dir_name}' is not a directory."}

        if item_to_remove.contents:
            return {"result": f"Error: Directory '{dir_name}' is not empty."}

        del self.current_dir.contents[dir_name]
        return {"result": f"Successfully removed directory '{dir_name}'."}

    def _deep_copy_item(
        self,
        item: Union[File, "Directory"],
        new_parent: "Directory",
        new_name: Optional[str] = None,
    ) -> Optional[Union[File, "Directory"]]:
        """Helper to recursively copy a file or directory."""
        name_to_use = new_name if new_name is not None else item.name
        if isinstance(item, File):
            return File(name=name_to_use, content=item.content, parent=new_parent)  # type: ignore[arg-type]
        elif isinstance(item, Directory):
            new_dir = Directory(name=name_to_use, parent=new_parent)
            new_contents: Dict[str, Union[File, "Directory"]] = {}
            for name_key, sub_item_val in item.contents.items():  # Iterate safely
                copied_sub_item = self._deep_copy_item(sub_item_val, new_dir)
                if copied_sub_item:  # Check if copy was successful
                    new_contents[name_key] = copied_sub_item
            new_dir.contents = new_contents
            return new_dir
        return None  # Should not happen if item is File or Directory

    def cp(self, source: str, destination: str) -> Dict:
        """Copy a file or directory."""
        # self.current_dir is Directory
        if source not in self.current_dir.contents:
            return {"result": f"Error: Source '{source}' not found."}

        source_item = self.current_dir.contents[
            source
        ]  # source_item is File or Directory

        dest_item_in_current_dir = self.current_dir.contents.get(destination)

        if isinstance(
            dest_item_in_current_dir, Directory
        ):  # Destination is an existing directory in current_dir
            dest_dir = dest_item_in_current_dir
            # source_item.name is safe as source_item is File or Directory
            if source_item.name in dest_dir.contents:
                existing_item_in_target = dest_dir.contents[source_item.name]
                if type(source_item) is not type(existing_item_in_target):
                    return {
                        "result": f"Error: Type mismatch. Cannot overwrite '{source_item.name}' in '{destination}'."
                    }
            copied_item = self._deep_copy_item(
                source_item, dest_dir, new_name=source_item.name
            )
            if copied_item:  # Check if copy was successful
                dest_dir.contents[copied_item.name] = (
                    copied_item  # copied_item.name is safe
                )
                return {"result": f"Copied '{source}' into directory '{destination}'."}
            else:
                return {"result": "Error: Failed to copy item."}  # Should be rare
        else:  # Destination is a new name or an existing file in current_dir
            if (
                destination in self.current_dir.contents
            ):  # Destination is an existing file
                existing_dest_item = self.current_dir.contents[destination]
                if isinstance(source_item, Directory) and isinstance(
                    existing_dest_item, File
                ):
                    return {
                        "result": f"Error: Cannot overwrite file '{destination}' with directory '{source}'."
                    }
                # Overwriting file with file, or dir with dir (if not caught by above as dest_dir)

            copied_item = self._deep_copy_item(
                source_item, self.current_dir, new_name=destination
            )
            if copied_item:  # Check if copy was successful
                self.current_dir.contents[destination] = copied_item
                return {"result": f"Copied '{source}' to '{destination}'."}
            else:
                return {"result": "Error: Failed to copy item."}  # Should be rare

    def ls(
        self, path: Optional[str] = None
    ) -> Dict:  # Overloaded method, keep original signature for now
        """List directory contents."""
        directory_to_list: Directory = self.current_dir
        if path:
            resolved_item = self._find_path(path)
            if not isinstance(resolved_item, Directory):
                return {"error": f"Path not found or not a directory: {path}"}
            directory_to_list = resolved_item

        # directory_to_list is Directory
        items: Dict[str, Dict[str, str]] = {}
        for name_key, item_val in directory_to_list.contents.items():
            if isinstance(item_val, Directory):
                items[name_key] = {"type": "directory"}
            elif isinstance(item_val, File):
                items[name_key] = {"type": "file"}
        return {"current_directory": directory_to_list.name, "contents": items}

    def cd(self, folder: str) -> Dict:  # Overloaded method
        """Change current directory."""
        # self.current_dir is Directory
        if folder == "..":
            parent_dir = self.current_dir.parent
            if parent_dir is not None:
                self.current_dir = parent_dir
                return {
                    "status": "success",
                    "message": f"Changed to {self.current_dir.name}",  # self.current_dir is Directory
                }
            return {"status": "error", "message": "Already at root directory"}

        target_item: Optional[Union[File, "Directory"]] = None
        if folder.startswith("/"):
            target_item = self._find_path(folder)
        elif folder in self.current_dir.contents:  # self.current_dir is Directory
            target_item = self.current_dir.contents[folder]

        if isinstance(target_item, Directory):  # Check if it's a Directory instance
            self.current_dir = target_item
            return {"status": "success", "message": f"Changed to {folder}"}

        return {
            "status": "error",
            "message": f"Directory '{folder}' not found or is not a directory.",
        }

    def mkdir(self, dir_name: str) -> Dict:  # Overloaded method
        """Create a new directory."""
        # self.current_dir is Directory
        if "/" in dir_name:
            return {"status": "error", "message": "Directory name cannot contain '/'."}
        if dir_name in self.current_dir.contents:
            return {
                "status": "error",
                "message": f"Directory '{dir_name}' already exists",
            }
        self.current_dir.contents[dir_name] = Directory(
            name=dir_name, parent=self.current_dir
        )
        return {"status": "success", "message": f"Created directory '{dir_name}'"}

    def cat(self, file_name: str) -> Dict:  # Overloaded method
        """Display file contents."""
        # self.current_dir is Directory
        item = self.current_dir.contents.get(file_name)
        if not isinstance(item, File):
            return {"error": f"File '{file_name}' not found"}
        # item is File
        return {
            "status": "success",
            "content": item.content,
        }

    def mv(self, source: str, destination: str) -> Dict:  # Overloaded method
        """Move a file or directory. Destination cannot be a path."""
        # self.current_dir is Directory
        if "/" in destination:
            return {"result": f"Error: Destination '{destination}' cannot be a path."}

        if source not in self.current_dir.contents:
            return {"result": f"Error: Source '{source}' not found."}

        source_item = self.current_dir.contents[source]  # File or Directory

        dest_as_dir_in_current = self.current_dir.contents.get(destination)

        if isinstance(
            dest_as_dir_in_current, Directory
        ):  # Moving into an existing directory
            target_dir = dest_as_dir_in_current
            if source == destination:  # Cannot move a dir into itself by the same name
                return {"result": f"Error: Cannot move '{source}' into itself."}

            # source_item.name is safe
            if source_item.name in target_dir.contents:
                existing_item_in_target = target_dir.contents[source_item.name]
                if type(source_item) is not type(existing_item_in_target):
                    return {
                        "result": f"Error: Type mismatch. Cannot overwrite '{source_item.name}' in '{destination}'."
                    }

            target_dir.contents[source_item.name] = source_item
            source_item.parent = target_dir  # source_item.parent is safe
            del self.current_dir.contents[source]
            return {"result": f"Moved '{source}' into directory '{destination}'."}
        else:  # Renaming or overwriting a file
            if (
                destination in self.current_dir.contents
            ):  # Destination is an existing file
                existing_dest_item = self.current_dir.contents[destination]
                if isinstance(source_item, Directory) and isinstance(
                    existing_dest_item, File
                ):
                    return {
                        "result": f"Error: Cannot overwrite file '{destination}' with directory '{source}'."
                    }

            self.current_dir.contents[destination] = source_item
            source_item.name = destination  # source_item.name is safe
            if source != destination:
                del self.current_dir.contents[source]
            return {"result": f"Moved '{source}' to '{destination}'."}

    def grep(self, file_name: str, pattern: str) -> Dict:  # Overloaded method
        """Search for lines in a file that contain the specified pattern."""
        # self.current_dir is Directory
        item = self.current_dir.contents.get(file_name)
        if not isinstance(item, File):
            return {"error": f"File '{file_name}' not found or is not a file."}

        # item is File
        content = item.content
        lines = content.splitlines()
        matching_lines = [line for line in lines if pattern in line]
        return {"matching_lines": matching_lines}

    def sort(self, file_name: str) -> Dict:  # Overloaded method
        """Sort the contents of a file line by line and return the sorted content."""
        # self.current_dir is Directory
        item = self.current_dir.contents.get(file_name)
        if not isinstance(item, File):
            return {"error": f"File '{file_name}' not found or is not a file."}

        # item is File
        content = item.content
        lines = content.splitlines()
        sorted_lines = sorted(lines)
        return {"sorted_content": "\n".join(sorted_lines)}

    def wc(self, file_name: str, mode: str = "l") -> Dict:
        """Count lines, words, or characters in a file."""
        # self.current_dir is Directory
        item = self.current_dir.contents.get(file_name)
        if not isinstance(item, File):
            return {"error": f"File '{file_name}' not found or is not a file."}

        # item is File
        content = item.content
        count = 0
        count_type_str = ""
        if mode == "l":
            count = len(content.splitlines())
            count_type_str = "lines"
        elif mode == "w":
            count = len(content.split())
            count_type_str = "words"
        elif mode == "c":
            count = len(content)
            count_type_str = "characters"
        else:
            return {"error": f"Invalid mode '{mode}'. Must be 'l', 'w', or 'c'."}
        return {"count": count, "type": count_type_str}

    def tail(self, file_name: str, lines: int = 10) -> Dict:
        """Display the last part of a file."""
        # self.current_dir is Directory
        item = self.current_dir.contents.get(file_name)
        if not isinstance(item, File):
            return {"error": f"File '{file_name}' not found or is not a file."}

        # item is File
        content = item.content
        content_lines = content.splitlines()
        if lines <= 0:
            last_n_lines = []
        else:
            last_n_lines = content_lines[-lines:]
        return {"last_lines": "\n".join(last_n_lines)}

    def _find_recursive(
        self,
        directory_obj: "Directory",
        search_name_pattern: Optional[str],
        current_relative_path: str,
        matches_list: list[str],
    ) -> None:
        """Helper for recursive find."""
        # directory_obj is Directory
        for name_key, item_val in directory_obj.contents.items():
            item_relative_path = f"{current_relative_path}{name_key}".lstrip("./")
            if (
                not item_relative_path
            ):  # Should not happen if current_relative_path or name_key is non-empty
                item_relative_path = name_key

            if (
                search_name_pattern is None
                or search_name_pattern.lower()
                == "none"  # Treat "none" string as no pattern
                or search_name_pattern in name_key  # name_key is str
            ):
                matches_list.append(item_relative_path)

            if isinstance(item_val, Directory):
                self._find_recursive(
                    item_val,
                    search_name_pattern,
                    f"{item_relative_path}/",
                    matches_list,
                )

    def find(
        self, path: str = ".", name: Optional[str] = None
    ) -> Dict:  # name can be None
        """Find files or directories recursively."""
        # self.current_dir is Directory
        start_dir_obj = self._find_path(path)

        if not isinstance(
            start_dir_obj, Directory
        ):  # Check if it's a Directory instance
            return {"error": f"Path '{path}' not found or is not a directory."}

        matches: list[str] = []  # Ensure matches is typed
        self._find_recursive(start_dir_obj, name, "", matches)
        return {"matches": matches}

    def _calculate_size_recursive(self, directory_obj: "Directory") -> int:
        """Helper to recursively calculate total size of files in a directory."""
        total_size = 0
        # directory_obj is Directory
        for item_val in directory_obj.contents.values():
            if isinstance(item_val, File):
                total_size += len(item_val.content)  # item_val.content is str
            elif isinstance(item_val, Directory):
                total_size += self._calculate_size_recursive(item_val)
        return total_size

    def _format_size_human_readable(self, size_bytes: int) -> str:
        """Converts size in bytes to human-readable string (KB, MB, GB)."""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.1f}KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / (1024**2):.1f}MB"
        else:
            return f"{size_bytes / (1024**3):.1f}GB"

    def du(self, human_readable: bool = False) -> Dict:
        """Estimate the disk usage of the current directory and its contents."""
        # self.current_dir is Directory
        total_size_bytes = self._calculate_size_recursive(self.current_dir)

        if human_readable:
            disk_usage_str = self._format_size_human_readable(total_size_bytes)
        else:
            disk_usage_str = str(total_size_bytes)

        return {"disk_usage": disk_usage_str}

    def echo(
        self, content: str, file_name: Optional[str] = None
    ) -> Dict:  # file_name can be None
        """Write content to a file or display it in the terminal."""
        # self.current_dir is Directory
        if file_name is None or file_name.lower() == "none":  # Explicit check for None
            return {"terminal_output": content}
        else:
            if "/" in file_name:
                return {"error": f"File name '{file_name}' cannot be a path."}

            item_in_dir = self.current_dir.contents.get(file_name)
            if isinstance(item_in_dir, Directory):  # Check if it's a Directory instance
                return {"error": f"Cannot write to '{file_name}': It is a directory."}

            self.current_dir.contents[file_name] = File(
                name=file_name, content=content, parent=self.current_dir
            )
            return {
                "terminal_output": None
            }  # Ensure None is returned, not an empty dict by mistake

    def diff(self, file_name1: str, file_name2: str) -> Dict:  # Overloaded method
        """Compare two files."""
        # self.current_dir is Directory
        item1 = self.current_dir.contents.get(file_name1)
        item2 = self.current_dir.contents.get(file_name2)

        if not isinstance(item1, File):
            return {"error": f"File {file_name1} not found"}
        if not isinstance(item2, File):
            return {"error": f"File {file_name2} not found"}

        # item1 and item2 are Files
        content1 = item1.content
        content2 = item2.content
        if content1 == content2:
            return {
                "status": "success",
                "diff_lines": "Files are identical",  # Match spec
            }
        else:
            lines1 = content1.split("\n")
            lines2 = content2.split("\n")
            diff_output_lines: list[str] = []  # Ensure type
            has_diff = False  # Not strictly needed if diff_output_lines is checked
            for i in range(max(len(lines1), len(lines2))):
                line1_val = lines1[i] if i < len(lines1) else None
                line2_val = lines2[i] if i < len(lines2) else None
                if line1_val != line2_val:
                    has_diff = True  # Redundant if checking diff_output_lines
                    l1_display = f"'{line1_val}'" if line1_val is not None else "None"
                    l2_display = f"'{line2_val}'" if line2_val is not None else "None"
                    diff_output_lines.append(
                        f"Line {i+1}: {l1_display} != {l2_display}"
                    )
            if not diff_output_lines:  # Check if any diffs were actually added
                return {"status": "success", "diff_lines": "Files are identical"}
            return {
                "status": "success",
                "diff_lines": "\n".join(diff_output_lines),
            }

    def _find_path(self, path_str: str) -> Optional[Union[File, "Directory"]]:
        """Helper to find an item (file or directory) from a path string."""
        if not self.root:
            return None

        current_item: Optional[Union[File, "Directory"]] = None
        parts_to_process: list[str] = []

        if path_str.startswith("/"):
            current_item = self.root
            normalized_path = path_str.lstrip("/")

            if not normalized_path:  # Path was "/"
                return self.root

            parts = normalized_path.split("/")

            if (
                parts and parts[0] == self.root.name
            ):  # Path like "/workspace" or "/workspace/dir"
                if len(parts) == 1:  # Path was "/workspace"
                    return self.root
                parts_to_process = parts[1:]  # Consume "workspace", process "dir"
            else:  # Path like "/dir" (interpreted as relative to root's contents)
                parts_to_process = parts
        else:  # Relative path
            current_item = self.current_dir
            if not path_str or path_str == ".":
                return self.current_dir
            parts_to_process = path_str.split("/")

        for part in parts_to_process:
            if not part:
                continue
            if part == "..":
                if current_item.parent:
                    current_item = current_item.parent
                else:
                    return None
            elif part == ".":
                continue
            if not isinstance(current_item, Directory):
                return None
            if part in current_item.contents:
                current_item = current_item.contents[part]
            else:
                return None

        return current_item
