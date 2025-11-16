from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SlurmHeader:
    """
    Container for SLURM '#SBATCH' directive settings.

    Each non-empty field in this dataclass generates a corresponding '#SBATCH' line in a job
    submission script. Use 'additional_lines' to include any custom directives not covered by
    the predefined attributes. Use 'shell_lines' to include non-#SBATCH shell lines (e.g.,
    'module load ...', exports) that should appear immediately after the #SBATCH block.
    """

    job_name: Optional[str] = None
    partition: Optional[str] = None
    nodes: Optional[int] = None
    ntasks: Optional[int] = None
    cpus_per_task: Optional[int] = None
    time: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None

    # Extra SBATCH lines like "--mem=60gb", "--hint=nomultithread", etc.
    additional_lines: List[str] = field(default_factory=list)

    # non-#SBATCH shell lines placed after the header (modules, exports, vars)
    shell_lines: List[str] = field(default_factory=list)

    def serialize(self) -> str:
        lines: List[str] = ["#!/bin/bash"]

        if self.job_name:
            lines.append(f"#SBATCH --job-name={self.job_name}")
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.nodes:
            lines.append(f"#SBATCH --nodes={self.nodes}")
        if self.ntasks:
            lines.append(f"#SBATCH --ntasks={self.ntasks}")
        if self.cpus_per_task:
            lines.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")
        if self.time:
            lines.append(f"#SBATCH --time={self.time}")
        if self.output:
            lines.append(f"#SBATCH --output={self.output}")
        if self.error:
            lines.append(f"#SBATCH --error={self.error}")

        for line in self.additional_lines:
            lines.append(line)

        if self.shell_lines:
            lines.append("")
            lines.extend(self.shell_lines)

        return "\n".join(lines)

    def to_string(self) -> str:
        """
        Backwards-compatible alias for serialize(), since SlurmFile.write()
        expects slurm_header.to_string().
        """
        return self.serialize()
