import os
import sys
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_PYTHON_VERSION = "3.10"


def _resolve_module_path_and_name(function_ref: str) -> Optional[Tuple[Path, str, str]]:
    """
    Resolves the file system path for the top-level module/package of a function reference
    and the module path to be copied.

    Args:
        function_ref: e.g., "my_package.my_module.my_function" or "my_module.my_function"

    Returns:
        A tuple (path_to_copy, top_level_module_name, relative_path_to_copy_as) if resolvable, else None.
        - path_to_copy: Absolute path to the directory or file to be copied.
        - top_level_module_name: The name of the top-level module/package (e.g., "my_package" or "my_module").
                                 This will be the destination name in the Docker image.
        - relative_path_to_copy_as: The path to use as the destination in the COPY instruction (e.g. "my_package" or "my_module.py")
    """
    parts = function_ref.split(".")
    if not parts:
        return None

    # Try to find the module/package in sys.path, prioritizing CWD
    # This is a simplified approach. A more robust one might involve inspecting __file__
    # of an imported module, but that requires the module to be importable in the current env.

    # Check CWD first
    potential_path_str = parts[0]  # e.g. "my_package" or "my_module"

    # Path to check in CWD
    path_in_cwd_dir = Path(os.getcwd()) / potential_path_str
    path_in_cwd_file = Path(os.getcwd()) / f"{potential_path_str}.py"

    if (
        path_in_cwd_dir.is_dir() and (path_in_cwd_dir / "__init__.py").exists()
    ):  # It's a package
        return path_in_cwd_dir, potential_path_str, potential_path_str
    elif path_in_cwd_file.is_file():  # It's a module .py file
        return path_in_cwd_file, potential_path_str, f"{potential_path_str}.py"

    # Fallback: could search sys.path but that gets complicated for Docker context.
    # For now, assume the module/package is in CWD or a sub-path accessible from CWD.
    # If function_ref is like "subdir.module.func", we assume "subdir" is in CWD.

    # If parts[0] is not directly in CWD, it might be a deeper structure.
    # This simplistic resolver assumes the first part of function_ref is the item to copy from CWD.
    # e.g. if function_ref is "app.services.rewards.my_func" and "app" is a dir in CWD.

    # A more robust solution for finding the "root" of the user's code might be needed,
    # or clear instructions on project structure.
    # For now, this handles simple cases: module.py in CWD or package/ in CWD.

    print(
        f"Warning: Could not reliably resolve path for '{function_ref}'. "
        f"Attempting to use '{parts[0]}' as the top-level module/package name from CWD."
    )

    # If we couldn't find it as a direct file or package, we can't be sure.
    # This part needs to be more robust or have clearer assumptions.
    # For now, let's assume if it's not found as above, it's an error for Dockerfile generation.
    return None


def generate_dockerfile_content(
    function_ref: str,
    python_version: str = DEFAULT_PYTHON_VERSION,
    reward_kit_install_source: str = "reward-kit",  # e.g., "reward-kit", "reward-kit[dev]", or path to local wheel/sdist
    user_requirements_path: Optional[str] = None,  # Path relative to CWD or absolute
    service_port: int = 8080,
) -> Optional[str]:
    """
    Generates the content for a Dockerfile to serve a given reward function.

    Args:
        function_ref: Python import string for the reward function (e.g., 'my_module.my_reward_func').
        python_version: The Python version for the base image (e.g., "3.10").
        reward_kit_install_source: Pip install string for reward-kit.
        user_requirements_path: Optional path to a requirements.txt for user dependencies.
        service_port: Port the service inside the container will listen on.

    Returns:
        The Dockerfile content as a string, or None if user code path cannot be resolved.
    """

    resolved_code = _resolve_module_path_and_name(function_ref)
    if not resolved_code:
        print(
            f"Error: Could not resolve path for function reference '{function_ref}'. "
            "Ensure the first part of the reference (e.g., 'my_module' or 'my_package') "
            "is a Python module (.py file) or package (directory with __init__.py) "
            "in the current working directory."
        )
        return None

    path_to_copy, top_module_name, copy_dest_name = resolved_code

    # Determine the source path for COPY relative to the Docker build context (assumed to be CWD)
    # If path_to_copy is absolute, we need its name relative to CWD for the COPY instruction.
    # For simplicity, _resolve_module_path_and_name returns paths that are effectively relative to CWD for copying.
    # So, copy_source_path will be top_module_name (for a package) or f"{top_module_name}.py" (for a module file).
    copy_source_path = copy_dest_name  # This is what we determined to copy (e.g. "my_package" or "my_module.py")

    dockerfile_lines = [
        f"FROM python:{python_version}-slim",
        "",
        "WORKDIR /app",
        "",
        "# Copy the entire application source (build context)",
        "COPY . .",  # Copies setup.py, reward_kit package, user's function module, etc.
        "",
        "# Install reward-kit from local source and its dependencies",
        # This assumes setup.py is configured to install reward_kit and its deps.
        # Add [dev] if extra dev dependencies are needed by generic_server itself, though unlikely.
        "RUN pip install --no-cache-dir .",
        "",
    ]

    # The user's reward function module (e.g., dummy_rewards.py) is now copied by "COPY . ."
    # So, the specific COPY for resolved_code is no longer needed here.
    # The function_ref in CMD will be resolved from /app.

    # Handle user-specific requirements.txt, if provided
    # This should be relative to the build context root (copied by "COPY . .")
    if user_requirements_path:
        user_req_file_in_context = Path(user_requirements_path)  # Path relative to CWD
        # Check if the file exists in the context (it would have been copied by "COPY . .")
        # For the RUN command, it will be at ./<user_requirements_path> if it was at the root of context
        # Or, if user_requirements_path is just "requirements.txt", it's simpler.
        # Let's assume user_requirements_path is the name of the file if it's in CWD.
        # If it's a path like "sub/requirements.txt", COPY . . handles it.

        # For simplicity, if user_requirements_path is e.g. "my_func_reqs.txt",
        # it would be copied to /app/my_func_reqs.txt.
        # The RUN pip install command needs to refer to it correctly.
        # If user_requirements_path is just "requirements.txt", it's fine.
        # If it's "path/to/requirements.txt", it will be at "/app/path/to/requirements.txt".

        # To make it robust, we should copy the requirements file to a standard name like "user_requirements.txt"
        # in the Dockerfile, or ensure the path is correctly referenced.
        # For now, let's assume user_requirements_path is a path relative to the build context root.
        # The COPY . . already copied it.

        # Check if the file specified by user_requirements_path exists at the root of the project
        # This check is done before Docker build by the CLI ideally.
        # Here, we just add the lines if the path is provided.
        dockerfile_lines.extend(
            [
                f"# Copy and install user-specific dependencies (if {user_requirements_path} exists in context)",
                # The file is already copied by "COPY . .". We just need to run pip install.
                # The path inside the container will be user_requirements_path relative to /app
                f'RUN if [ -f {user_requirements_path} ]; then pip install --no-cache-dir -r {user_requirements_path}; else echo "User requirements file {user_requirements_path} not found in /app, skipping."; fi',
                "",
            ]
        )

    dockerfile_lines.extend(
        [
            f"ENV PORT {service_port}",
            f"EXPOSE {service_port}",
            "",
            "# Run the generic server, pointing to the user's function",
            # Using shell form for CMD to allow $PORT expansion
            f"CMD python -m reward_kit.generic_server {function_ref} --host 0.0.0.0 --port $PORT",
        ]
    )

    return "\n".join(dockerfile_lines)


if __name__ == "__main__":
    # Example Usage
    print("--- Example 1: Module in CWD ---")
    # Create dummy my_test_reward_module.py
    with open("my_test_reward_module.py", "w") as f:
        f.write("def my_reward_func(): pass\n")

    dockerfile_content_module = generate_dockerfile_content(
        function_ref="my_test_reward_module.my_reward_func",
        user_requirements_path="dummy_requirements.txt",  # Test non-existent req file
    )
    if dockerfile_content_module:
        print("\nGenerated Dockerfile (module):")
        print(dockerfile_content_module)
    os.remove("my_test_reward_module.py")

    print("\n--- Example 2: Package in CWD ---")
    # Create dummy package my_test_reward_pkg/
    pkg_name = "my_test_reward_pkg"
    Path(pkg_name).mkdir(exist_ok=True)
    with open(Path(pkg_name) / "__init__.py", "w") as f:
        f.write("# Package init\n")
    with open(Path(pkg_name) / "rewards.py", "w") as f:
        f.write("def complex_reward_func(): pass\n")

    # Create dummy requirements.txt for this package example
    user_reqs_name = "pkg_requirements.txt"
    with open(user_reqs_name, "w") as f:
        f.write("numpy==1.23.0\n")

    dockerfile_content_pkg = generate_dockerfile_content(
        function_ref="my_test_reward_pkg.rewards.complex_reward_func",
        user_requirements_path=user_reqs_name,
    )
    if dockerfile_content_pkg:
        print("\nGenerated Dockerfile (package):")
        print(dockerfile_content_pkg)

    # Cleanup
    os.remove(Path(pkg_name) / "rewards.py")
    os.remove(Path(pkg_name) / "__init__.py")
    Path(pkg_name).rmdir()
    os.remove(user_reqs_name)

    print("\n--- Example 3: Unresolvable function ref ---")
    dockerfile_content_bad = generate_dockerfile_content(
        function_ref="non_existent_module.some_func"
    )
    if dockerfile_content_bad:
        print(dockerfile_content_bad)
    else:
        print("Dockerfile generation failed as expected for non_existent_module.")
