import os

from typing import List


def get_package_xmls() -> List[str]:
    """Returns a list of package.xml files."""
    # Get the path to the models directory relative to this file
    util_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(util_dir))
    path = os.path.join(repo_root, "models", "package.xml")

    if os.path.exists(path):
        return [path]
    return []
