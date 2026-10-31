from __future__ import annotations

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

from ..config import ProjectPaths


class WorkspaceBackendBuilder:
    """Exposes ONLY data/, artifacts/, plots/ and skills/ to the agent's file tools.

    src/ and docker/ are deliberately unreachable. Anything written elsewhere lands in
    ephemeral agent state (scratch space, e.g. notes), never on disk.
    """

    def __init__(self, paths: ProjectPaths) -> None:
        self._paths = paths

    def build(self) -> CompositeBackend:
        def fs(root):  # virtual_mode confines every path to root
            return FilesystemBackend(root_dir=str(root), virtual_mode=True)

        return CompositeBackend(
            default=StateBackend(),
            routes={
                "/data/": fs(self._paths.data),
                "/artifacts/": fs(self._paths.artifacts),
                "/plots/": fs(self._paths.plots),
                "/skills/": fs(self._paths.skills),
            },
        )
