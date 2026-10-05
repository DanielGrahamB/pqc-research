# Template embedded by scripts/build_colab_bundle.py; run the notebook cell.
# @title Colab setup: unpack the bundled project and install dependencies
import base64
import hashlib
import importlib
import importlib.metadata
import io
import os
from pathlib import Path
import subprocess
import sys
import zipfile

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Set this before importing torch or initializing CUDA for deterministic matmul.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
BUNDLE_SHA256 = "__BUNDLE_SHA256__"
BUNDLE_BASE64 = "__BUNDLE_BASE64__"

if IN_COLAB:
    if sys.version_info < (3, 11):
        raise RuntimeError("Sionna 2.0.1 requires Python 3.11+. Select a current Colab runtime.")
    archive = base64.b64decode(BUNDLE_BASE64)
    if hashlib.sha256(archive).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError("Embedded project checksum mismatch; upload a fresh notebook.")
    PROJECT_ROOT = Path("/content/pqc-project") / BUNDLE_SHA256[:16]
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    # A rerun verifies existing files; it never overwrites edits or results.
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for member in bundle.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts or member.is_dir():
                raise RuntimeError(f"Unexpected bundle member: {member.filename}")
            target = PROJECT_ROOT / relative
            data = bundle.read(member)
            if target.exists() and target.read_bytes() != data:
                raise RuntimeError(f"Bundled source was edited: {target}. Preserve your edits and use a fresh runtime or rebuild the bundle.")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)
    active = sys.modules.get("pqc_experiments")
    if active is not None and Path(active.__file__).resolve().parent != PROJECT_ROOT / "pqc_experiments":
        raise RuntimeError("A different project version is imported. Restart the Colab session, then Run all.")
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from packaging.requirements import Requirement
    requirements = [Requirement(line) for line in Path("requirements-colab.txt").read_text().splitlines()
                    if line.strip() and not line.startswith("#")]
    def installed_version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    missing = [str(req) for req in requirements
               if installed_version(req.name) is None or not req.specifier.contains(installed_version(req.name))]
    if missing:
        # Record loaded numerical libraries so a pip upgrade cannot silently leave
        # old modules in memory while reporting new on-disk package versions.
        loaded = {req.name: installed_version(req.name) for req in requirements if req.name in sys.modules}
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                               "-r", str(PROJECT_ROOT / "requirements-colab.txt")])
        importlib.invalidate_caches()
        changed = [name for name, version in loaded.items() if installed_version(name) != version]
        if changed:
            raise RuntimeError("Installed updated libraries already loaded in memory: " + ", ".join(changed)
                               + ". Use Runtime > Restart session, then Run all again. Project files are already unpacked.")
    print(f"Project ready: {PROJECT_ROOT}")
    print("Bundle SHA256:", BUNDLE_SHA256)
else:
    PROJECT_ROOT = Path.cwd()
    if not (PROJECT_ROOT / "pqc_experiments").is_dir():
        raise RuntimeError("For local execution, start Jupyter in the ps-layer-security repository directory.")
    print("Local execution: using repository files; dependencies were not changed.")
