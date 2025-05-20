"""Implementation of GorillaFileSystem."""


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
        if "root" in config:
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

    def ls(self, path=None):
        """List directory contents."""
        directory = self.current_dir
        if path:
            directory = self._find_path(path)
            if not directory or not isinstance(directory, Directory):
                return {"error": f"Path not found or not a directory: {path}"}

        items = {}
        for name, item in directory.contents.items():
            if isinstance(item, Directory):
                items[name] = {"type": "directory"}
            elif isinstance(item, File):
                items[name] = {"type": "file"}

        return {"current_directory": directory.name, "contents": items}

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

        if folder in self.current_dir.contents and isinstance(
            self.current_dir.contents[folder], Directory
        ):
            self.current_dir = self.current_dir.contents[folder]
            return {"status": "success", "message": f"Changed to {folder}"}

        return {"status": "error", "message": f"Directory {folder} not found"}

    def mkdir(self, dir_name):
        """Create a new directory."""
        if dir_name in self.current_dir.contents:
            return {
                "status": "error",
                "message": f"Directory {dir_name} already exists",
            }

        self.current_dir.contents[dir_name] = Directory(
            name=dir_name, parent=self.current_dir
        )
        return {"status": "success", "message": f"Created directory {dir_name}"}

    def cat(self, file_name):
        """Display file contents."""
        if file_name not in self.current_dir.contents or not isinstance(
            self.current_dir.contents[file_name], File
        ):
            return {"status": "error", "message": f"File {file_name} not found"}

        return {
            "status": "success",
            "content": self.current_dir.contents[file_name].content,
        }

    def mv(self, source, destination):
        """Move a file or directory."""
        # Simple implementation for the mock
        if source not in self.current_dir.contents:
            return {"status": "error", "message": f"Source {source} not found"}

        parts = destination.split("/")
        if len(parts) == 1:
            # Same directory, just rename
            self.current_dir.contents[destination] = self.current_dir.contents[source]
            self.current_dir.contents[destination].name = destination
            del self.current_dir.contents[source]
            return {"status": "success", "message": f"Moved {source} to {destination}"}
        else:
            # Moving to another directory
            target_dir_name = parts[0]
            if target_dir_name not in self.current_dir.contents or not isinstance(
                self.current_dir.contents[target_dir_name], Directory
            ):
                return {
                    "status": "error",
                    "message": f"Target directory {target_dir_name} not found",
                }

            target_dir = self.current_dir.contents[target_dir_name]
            target_dir.contents[source] = self.current_dir.contents[source]
            target_dir.contents[source].parent = target_dir
            del self.current_dir.contents[source]
            return {
                "status": "success",
                "message": f"Moved {source} to {target_dir_name}/{source}",
            }

    def grep(self, file_name, pattern):
        """Search for a pattern in a file."""
        if file_name not in self.current_dir.contents or not isinstance(
            self.current_dir.contents[file_name], File
        ):
            return {"status": "error", "message": f"File {file_name} not found"}

        content = self.current_dir.contents[file_name].content
        lines = content.split("\n")
        matches = [line for line in lines if pattern in line]

        return {"status": "success", "matches": matches, "count": len(matches)}

    def sort(self, file_name):
        """Sort the lines in a file."""
        if file_name not in self.current_dir.contents or not isinstance(
            self.current_dir.contents[file_name], File
        ):
            return {"status": "error", "message": f"File {file_name} not found"}

        content = self.current_dir.contents[file_name].content
        lines = content.split("\n")
        sorted_lines = sorted(lines)

        self.current_dir.contents[file_name].content = "\n".join(sorted_lines)

        return {"status": "success", "message": f"Sorted {file_name}"}

    def diff(self, file_name1, file_name2):
        """Compare two files."""
        if file_name1 not in self.current_dir.contents or not isinstance(
            self.current_dir.contents[file_name1], File
        ):
            return {"status": "error", "message": f"File {file_name1} not found"}

        if file_name2 not in self.current_dir.contents or not isinstance(
            self.current_dir.contents[file_name2], File
        ):
            return {"status": "error", "message": f"File {file_name2} not found"}

        content1 = self.current_dir.contents[file_name1].content
        content2 = self.current_dir.contents[file_name2].content

        if content1 == content2:
            return {
                "status": "success",
                "message": "Files are identical",
                "differences": [],
            }
        else:
            # Simple line-by-line diff
            lines1 = content1.split("\n")
            lines2 = content2.split("\n")

            differences = []
            for i in range(max(len(lines1), len(lines2))):
                line1 = lines1[i] if i < len(lines1) else None
                line2 = lines2[i] if i < len(lines2) else None

                if line1 != line2:
                    differences.append({"line": i + 1, "file1": line1, "file2": line2})

            return {
                "status": "success",
                "message": f"Found {len(differences)} differences",
                "differences": differences,
            }

    def _find_path(self, path):
        """Helper to find a path in the file system."""
        if path.startswith("/"):
            current = self.root
            path = path[1:]
        else:
            current = self.current_dir

        if not path:
            return current

        parts = path.split("/")
        for part in parts:
            if part == "..":
                if current.parent:
                    current = current.parent
                else:
                    return None
            elif part == ".":
                continue
            elif part in current.contents and isinstance(
                current.contents[part], Directory
            ):
                current = current.contents[part]
            else:
                return None

        return current
