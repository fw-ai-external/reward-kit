"""Implementation of GorillaFileSystem for BFCL evaluation."""


class File:
    """A file in the Gorilla File System."""

    def __init__(self, name="", content="", parent=None):
        self.name = name
        self.content = content
        self.parent = parent

    def __repr__(self):
        return f"<File: {self.name}, Content: {self.content}>"

    def __eq__(self, other):
        if not isinstance(other, File):
            return False
        return self.name == other.name and self.content == other.content


class Directory:
    """A directory in the Gorilla File System."""

    def __init__(self, name="", parent=None, contents=None):
        self.name = name
        self.parent = parent
        self.contents = contents or {}

    def __repr__(self):
        return f"<Directory: {self.name}, Parent: {self.parent.name if self.parent else None}, Contents: {self.contents}>"

    def __eq__(self, other):
        if not isinstance(other, Directory):
            return False
        return self.name == other.name and self.contents == other.contents


class GorillaFileSystem:
    """A file system for BFCL evaluation."""

    def __init__(self):
        self.root = None
        self.current_dir = None
        self.long_context = False

    def _load_scenario(self, config):
        """Load the file system from configuration."""
        if (
            not config or "root" not in config
        ):  # Ensure root is created for empty or no-root config
            self.root = Directory(name="workspace", parent=None)
            self.current_dir = self.root
        elif "root" in config:
            try:
                # Try to load using standard format
                if isinstance(config["root"], dict) and "type" in config["root"]:
                    self.root = self._load_directory_from_config(
                        "workspace", None, config["root"]
                    )
                # Try to load using the format in multi_turn_base_0.yaml
                else:
                    self.root = self._load_directory_from_yaml_config(
                        "workspace", None, config["root"]
                    )
                self.current_dir = self.root
            except Exception as e:
                print(f"Error loading GorillaFileSystem scenario: {e}")
                # Create a minimal root directory as fallback
                self.root = Directory(name="workspace", parent=None)
                self.current_dir = self.root

        if "long_context" in config:
            self.long_context = config["long_context"]

    def _load_directory_from_config(self, name, parent, config):
        """Create a directory structure from configuration."""
        if config["type"] == "directory":
            directory = Directory(name=name, parent=parent)
            contents = {}
            for item_name, item_config in config.get("contents", {}).items():
                if item_config["type"] == "directory":
                    contents[item_name] = self._load_directory_from_config(
                        item_name, directory, item_config
                    )
                elif item_config["type"] == "file":
                    contents[item_name] = File(
                        name=item_name, content=item_config["content"], parent=directory
                    )
            directory.contents = contents
            return directory
        return None

    def _load_directory_from_yaml_config(self, name, parent, config):
        """Create a directory structure from YAML configuration format."""
        directory = Directory(name=name, parent=parent)
        contents = {}

        if "contents" in config:
            for item_name, item_config in config["contents"].items():
                if isinstance(item_config, dict) and "contents" in item_config:
                    # This is a directory
                    contents[item_name] = self._load_directory_from_yaml_config(
                        item_name, directory, item_config
                    )
                elif isinstance(item_config, dict) and "content" in item_config:
                    # This is a file
                    contents[item_name] = File(
                        name=item_name, content=item_config["content"], parent=directory
                    )
                elif (
                    isinstance(item_config, dict)
                    and "type" in item_config
                    and item_config["type"] == "directory"
                ):
                    # This is explicitly a directory
                    contents[item_name] = self._load_directory_from_yaml_config(
                        item_name, directory, item_config
                    )
                elif (
                    isinstance(item_config, dict)
                    and "type" in item_config
                    and item_config["type"] == "file"
                ):
                    # This is explicitly a file
                    contents[item_name] = File(
                        name=item_name,
                        content=item_config.get("content", ""),
                        parent=directory,
                    )

        directory.contents = contents
        return directory

    def pwd(self):
        """Return the current working directory path."""
        if not self.current_dir:
            return {"error": "File system not initialized."}
        if not self.root:  # Should always have a root if initialized
            return {"error": "Root directory not found."}

        path_parts = []
        temp_dir = self.current_dir

        # Traverse up to the root
        while temp_dir and temp_dir != self.root:
            path_parts.append(temp_dir.name)
            temp_dir = temp_dir.parent
            if (
                temp_dir is None and self.current_dir != self.root
            ):  # Safety break if parent is None unexpectedly
                return {
                    "error": "File system structure error: encountered None parent before root."
                }

        path_parts.append(self.root.name)  # Add root's name

        # Construct the path string (e.g., /workspace/dir1/dir2)
        # The list is reversed because we traversed from current up to root.
        full_path = "/" + "/".join(reversed(path_parts))
        return {"current_working_directory": full_path}

    def touch(self, file_name: str):
        """Create a new empty file or update timestamp (noop for content)."""
        if not self.current_dir:
            return {
                "error": "File system not initialized."
            }  # Or specific response based on spec

        if file_name in self.current_dir.contents:
            if isinstance(self.current_dir.contents[file_name], Directory):
                return {"error": f"Cannot touch '{file_name}': It is a directory."}
            # File exists, standard touch updates timestamp, here we do nothing to content
            return {}  # Success, empty response as per spec

        # Create new empty file
        self.current_dir.contents[file_name] = File(
            name=file_name, content="", parent=self.current_dir
        )
        return {}  # Success, empty response as per spec

    def rm(self, file_name: str):
        """Remove a file or directory."""
        if not self.current_dir:
            return {"result": "Error: File system not initialized."}

        if file_name not in self.current_dir.contents:
            return {"result": f"Error: '{file_name}' not found."}

        del self.current_dir.contents[file_name]
        return {"result": f"Successfully removed '{file_name}'."}

    def rmdir(self, dir_name: str):
        """Remove an empty directory."""
        if not self.current_dir:
            return {"result": "Error: File system not initialized."}

        if dir_name not in self.current_dir.contents:
            return {"result": f"Error: Directory '{dir_name}' not found."}

        item_to_remove = self.current_dir.contents[dir_name]

        if not isinstance(item_to_remove, Directory):
            return {"result": f"Error: '{dir_name}' is not a directory."}

        if item_to_remove.contents:  # Check if directory is empty
            return {"result": f"Error: Directory '{dir_name}' is not empty."}

        del self.current_dir.contents[dir_name]
        return {"result": f"Successfully removed directory '{dir_name}'."}

    def _deep_copy_item(self, item, new_parent, new_name=None):
        """Helper to recursively copy a file or directory."""
        name_to_use = new_name if new_name is not None else item.name
        if isinstance(item, File):
            return File(name=name_to_use, content=item.content, parent=new_parent)
        elif isinstance(item, Directory):
            new_dir = Directory(name=name_to_use, parent=new_parent)
            new_contents = {}
            for name, sub_item in item.contents.items():
                new_contents[name] = self._deep_copy_item(sub_item, new_dir)
            new_dir.contents = new_contents
            return new_dir
        return None

    def cp(self, source: str, destination: str):
        """Copy a file or directory."""
        if not self.current_dir:
            return {"result": "Error: File system not initialized."}

        if source not in self.current_dir.contents:
            return {"result": f"Error: Source '{source}' not found."}

        source_item = self.current_dir.contents[source]

        if destination in self.current_dir.contents and isinstance(
            self.current_dir.contents[destination], Directory
        ):
            dest_dir = self.current_dir.contents[destination]
            if source_item.name in dest_dir.contents:
                existing_item_in_target = dest_dir.contents[source_item.name]
                if type(source_item) is not type(existing_item_in_target):
                    return {
                        "result": f"Error: Type mismatch. Cannot overwrite '{source_item.name}' in '{destination}'."
                    }
            copied_item = self._deep_copy_item(
                source_item, dest_dir, new_name=source_item.name
            )  # Use source_item.name
            if copied_item:
                dest_dir.contents[copied_item.name] = copied_item
                return {"result": f"Copied '{source}' into directory '{destination}'."}
            else:
                return {"result": "Error: Failed to copy item."}
        else:
            if destination in self.current_dir.contents:
                existing_dest_item = self.current_dir.contents[destination]
                if isinstance(source_item, Directory) and isinstance(
                    existing_dest_item, File
                ):
                    return {
                        "result": f"Error: Cannot overwrite file '{destination}' with directory '{source}'."
                    }
                if isinstance(source_item, File) and isinstance(
                    existing_dest_item, Directory
                ):
                    return {
                        "result": f"Error: Cannot overwrite directory '{destination}' with file '{source}'."
                    }
            copied_item = self._deep_copy_item(
                source_item, self.current_dir, new_name=destination
            )
            if copied_item:
                self.current_dir.contents[destination] = copied_item
                return {"result": f"Copied '{source}' to '{destination}'."}
            else:
                return {"result": "Error: Failed to copy item."}

    def ls(self, path=None):
        """List directory contents."""
        directory_to_list = self.current_dir
        if path:
            resolved_item = self._find_path(path)
            if not resolved_item or not isinstance(resolved_item, Directory):
                return {"error": f"Path not found or not a directory: {path}"}
            directory_to_list = resolved_item

        items = {}
        for name, item in directory_to_list.contents.items():
            if isinstance(item, Directory):
                items[name] = {"type": "directory"}
            elif isinstance(item, File):
                items[name] = {"type": "file"}
        return {"current_directory": directory_to_list.name, "contents": items}

    def cd(self, folder):
        """Change current directory."""
        if folder == "..":
            if self.current_dir.parent:
                self.current_dir = self.current_dir.parent
                return {
                    "status": "success",
                    "message": f"Changed to {self.current_dir.name}",
                }
            return {"status": "error", "message": "Already at root directory"}

        target_item = None
        if folder.startswith("/"):
            target_item = self._find_path(folder)
        elif folder in self.current_dir.contents:
            target_item = self.current_dir.contents[folder]

        if target_item and isinstance(target_item, Directory):
            self.current_dir = target_item
            return {"status": "success", "message": f"Changed to {folder}"}

        return {
            "status": "error",
            "message": f"Directory '{folder}' not found or is not a directory.",
        }

    def mkdir(self, dir_name):
        """Create a new directory."""
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

    def cat(self, file_name):
        """Display file contents."""
        if file_name not in self.current_dir.contents or not isinstance(
            self.current_dir.contents[file_name], File
        ):
            return {"error": f"File '{file_name}' not found"}
        return {
            "status": "success",
            "content": self.current_dir.contents[file_name].content,
        }

    def mv(self, source: str, destination: str):
        """Move a file or directory. Destination cannot be a path."""
        if not self.current_dir:
            return {"result": "Error: File system not initialized."}

        if "/" in destination:
            return {"result": f"Error: Destination '{destination}' cannot be a path."}

        if source not in self.current_dir.contents:
            return {"result": f"Error: Source '{source}' not found."}

        source_item = self.current_dir.contents[source]

        if destination in self.current_dir.contents and isinstance(
            self.current_dir.contents[destination], Directory
        ):
            target_dir = self.current_dir.contents[destination]
            if source == destination:
                return {"result": f"Error: Cannot move '{source}' into itself."}
            if source_item.name in target_dir.contents:
                existing_item_in_target = target_dir.contents[source_item.name]
                if type(source_item) is not type(existing_item_in_target):
                    return {
                        "result": f"Error: Type mismatch. Cannot overwrite '{source_item.name}' in '{destination}'."
                    }
            target_dir.contents[source_item.name] = source_item
            source_item.parent = target_dir
            del self.current_dir.contents[source]
            return {"result": f"Moved '{source}' into directory '{destination}'."}
        else:
            if destination in self.current_dir.contents:
                existing_dest_item = self.current_dir.contents[destination]
                if isinstance(source_item, Directory) and isinstance(
                    existing_dest_item, File
                ):
                    return {
                        "result": f"Error: Cannot overwrite file '{destination}' with directory '{source}'."
                    }
                if isinstance(source_item, File) and isinstance(
                    existing_dest_item, Directory
                ):
                    return {
                        "result": f"Error: Cannot overwrite directory '{destination}' with file '{source}' (unexpected; should be caught by previous block)."
                    }
            self.current_dir.contents[destination] = source_item
            source_item.name = destination
            if source != destination:
                del self.current_dir.contents[source]
            return {"result": f"Moved '{source}' to '{destination}'."}

    def grep(self, file_name: str, pattern: str):
        """Search for lines in a file that contain the specified pattern."""
        if not self.current_dir:
            return {"error": "File system not initialized."}
        if file_name not in self.current_dir.contents or not isinstance(
            self.current_dir.contents[file_name], File
        ):
            return {"error": f"File '{file_name}' not found or is not a file."}
        content = self.current_dir.contents[file_name].content
        lines = content.splitlines()
        matching_lines = [line for line in lines if pattern in line]
        return {"matching_lines": matching_lines}

    def sort(self, file_name: str):
        """Sort the contents of a file line by line and return the sorted content."""
        if not self.current_dir:
            return {"error": "File system not initialized."}
        if file_name not in self.current_dir.contents or not isinstance(
            self.current_dir.contents[file_name], File
        ):
            return {"error": f"File '{file_name}' not found or is not a file."}
        content = self.current_dir.contents[file_name].content
        lines = content.splitlines()
        sorted_lines = sorted(lines)
        return {"sorted_content": "\n".join(sorted_lines)}

    def wc(self, file_name: str, mode: str = "l"):
        """Count lines, words, or characters in a file."""
        if not self.current_dir:
            return {"error": "File system not initialized."}
        if file_name not in self.current_dir.contents or not isinstance(
            self.current_dir.contents[file_name], File
        ):
            return {"error": f"File '{file_name}' not found or is not a file."}
        file_item = self.current_dir.contents[file_name]
        content = file_item.content
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

    def tail(self, file_name: str, lines: int = 10):
        """Display the last part of a file."""
        if not self.current_dir:
            return {"error": "File system not initialized."}
        if file_name not in self.current_dir.contents or not isinstance(
            self.current_dir.contents[file_name], File
        ):
            return {"error": f"File '{file_name}' not found or is not a file."}
        file_item = self.current_dir.contents[file_name]
        content = file_item.content
        content_lines = content.splitlines()
        if lines <= 0:
            last_n_lines = []
        else:
            last_n_lines = content_lines[-lines:]
        return {"last_lines": "\n".join(last_n_lines)}

    def _find_recursive(
        self, directory_obj, search_name_pattern, current_relative_path, matches_list
    ):
        """Helper for recursive find."""
        for name, item in directory_obj.contents.items():
            item_relative_path = f"{current_relative_path}{name}".lstrip("./")
            if not item_relative_path:
                item_relative_path = name

            if (
                search_name_pattern is None
                or search_name_pattern.lower() == "none"
                or search_name_pattern in name
            ):
                matches_list.append(item_relative_path)

            if isinstance(item, Directory):
                self._find_recursive(
                    item, search_name_pattern, f"{item_relative_path}/", matches_list
                )

    def find(self, path: str = ".", name: str = None):
        """Find files or directories recursively."""
        if not self.current_dir:
            return {"error": "File system not initialized."}

        start_dir_obj = self._find_path(path)  # Use the corrected _find_path

        if not start_dir_obj or not isinstance(start_dir_obj, Directory):
            return {"error": f"Path '{path}' not found or is not a directory."}

        matches = []
        self._find_recursive(start_dir_obj, name, "", matches)
        return {"matches": matches}

    def _calculate_size_recursive(self, directory_obj: Directory) -> int:
        """Helper to recursively calculate total size of files in a directory."""
        total_size = 0
        for item in directory_obj.contents.values():
            if isinstance(item, File):
                total_size += len(item.content)
            elif isinstance(item, Directory):
                total_size += self._calculate_size_recursive(item)
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

    def du(self, human_readable: bool = False):
        """Estimate the disk usage of the current directory and its contents."""
        if not self.current_dir:
            return {"error": "File system not initialized."}

        # The spec implies du on current directory. If a path were allowed:
        # start_dir_obj = self._find_path(path_param_if_any)
        # if not start_dir_obj or not isinstance(start_dir_obj, Directory):
        #     return {"error": f"Path '{path_param_if_any}' not found or not a directory."}
        # total_size_bytes = self._calculate_size_recursive(start_dir_obj)

        total_size_bytes = self._calculate_size_recursive(self.current_dir)

        if human_readable:
            disk_usage_str = self._format_size_human_readable(total_size_bytes)
        else:
            disk_usage_str = str(total_size_bytes)  # As bytes

        return {"disk_usage": disk_usage_str}

    def echo(self, content: str, file_name: str = None):
        """Write content to a file or display it in the terminal."""
        if not self.current_dir:
            return {"error": "File system not initialized."}
        if file_name is None or file_name.lower() == "none":
            return {"terminal_output": content}
        else:
            if "/" in file_name:
                return {"error": f"File name '{file_name}' cannot be a path."}
            if file_name in self.current_dir.contents and isinstance(
                self.current_dir.contents[file_name], Directory
            ):
                return {"error": f"Cannot write to '{file_name}': It is a directory."}
            self.current_dir.contents[file_name] = File(
                name=file_name, content=content, parent=self.current_dir
            )
            return {"terminal_output": None}

    def diff(self, file_name1, file_name2):
        """Compare two files."""
        if file_name1 not in self.current_dir.contents or not isinstance(
            self.current_dir.contents[file_name1], File
        ):
            return {"error": f"File {file_name1} not found"}
        if file_name2 not in self.current_dir.contents or not isinstance(
            self.current_dir.contents[file_name2], File
        ):
            return {"error": f"File {file_name2} not found"}
        content1 = self.current_dir.contents[file_name1].content
        content2 = self.current_dir.contents[file_name2].content
        if content1 == content2:
            return {
                "status": "success",
                "diff_lines": "Files are identical",
            }
        else:
            lines1 = content1.split("\n")
            lines2 = content2.split("\n")
            diff_output_lines = []
            has_diff = False
            for i in range(max(len(lines1), len(lines2))):
                line1 = lines1[i] if i < len(lines1) else None
                line2 = lines2[i] if i < len(lines2) else None
                if line1 != line2:
                    has_diff = True
                    l1_display = f"'{line1}'" if line1 is not None else "None"
                    l2_display = f"'{line2}'" if line2 is not None else "None"
                    diff_output_lines.append(
                        f"Line {i+1}: {l1_display} != {l2_display}"
                    )
            if not has_diff:
                return {"status": "success", "diff_lines": "Files are identical"}
            return {
                "status": "success",
                "diff_lines": "\n".join(diff_output_lines),
            }

    def _find_path(self, path_str: str):
        """Helper to find an item (file or directory) from a path string."""
        if not self.root:
            return None

        current_item = None
        parts_to_process = []

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
