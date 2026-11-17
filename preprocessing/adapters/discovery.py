"""Schema-driven case discovery. Layouts are verified against local files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from preprocessing.errors import (
    DatasetLayoutError,
    DatasetNotFoundError,
    MissingModalityError,
)
from preprocessing.io_utils import CANONICAL_MODALITIES
from training.config import project_root, resolve_path


@dataclass
class CaseRecord:
    patient_id: str
    dataset: str
    modalities: dict[str, Path]
    segmentation: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def missing_required(self) -> list[str]:
        return [name for name in CANONICAL_MODALITIES if name not in self.modalities and self.extra.get("required", {}).get(name, False)]


@dataclass
class DiscoveryReport:
    dataset: str
    root: Path
    cases: list[CaseRecord]
    incomplete: list[dict[str, Any]]
    unmatched_files: list[str]

    @property
    def complete_cases(self) -> list[CaseRecord]:
        incomplete_ids = {item["patient_id"] for item in self.incomplete}
        return [case for case in self.cases if case.patient_id not in incomplete_ids]


def _match(filename: str, patterns: list[str]) -> bool:
    from fnmatch import fnmatch

    name = filename.lower()
    return any(fnmatch(name, pattern.lower()) for pattern in patterns)


def discover(schema: dict[str, Any]) -> DiscoveryReport:
    root = resolve_path(schema["root"], project_root())
    if not root.exists():
        raise DatasetNotFoundError(
            f"{schema['name']} root does not exist: {root}. "
            "Download the dataset and point root at the local files. "
            "This schema is not evidence that the files are present."
        )
    if schema.get("format") == "dicom":
        return _discover_dicom(schema, root)

    required = {
        name: bool(spec.get("required", False))
        for name, spec in schema.get("modalities", {}).items()
    }
    nifti_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))
    ]
    if not nifti_files:
        raise DatasetLayoutError(
            f"No NIfTI files under {root}. The configured patterns were not applied "
            "because there is nothing to match. Inspect the download layout."
        )

    grouped: dict[str, dict[str, Path]] = {}
    seg_paths: dict[str, Path] = {}
    matched: set[Path] = set()
    seg_patterns = list(schema.get("labels", {}).get("segmentation", {}).get("patterns") or [])

    for path in nifti_files:
        patient_id = path.parent.name if schema.get("patient_id_from") == "parent_directory" else path.parent.name
        assigned = False
        for modality, spec in schema.get("modalities", {}).items():
            if _match(path.name, list(spec.get("patterns") or [])):
                slot = grouped.setdefault(patient_id, {})
                if modality in slot:
                    raise DatasetLayoutError(
                        f"Multiple files matched {modality} for {patient_id}: "
                        f"{slot[modality].name} and {path.name}. Tighten the patterns."
                    )
                slot[modality] = path
                matched.add(path)
                assigned = True
                break
        if not assigned and seg_patterns and _match(path.name, seg_patterns):
            if patient_id in seg_paths:
                raise DatasetLayoutError(
                    f"Multiple segmentation files for {patient_id}: "
                    f"{seg_paths[patient_id].name} and {path.name}."
                )
            seg_paths[patient_id] = path
            matched.add(path)

    cases: list[CaseRecord] = []
    incomplete: list[dict[str, Any]] = []
    for patient_id in sorted(set(grouped) | set(seg_paths)):
        modalities = grouped.get(patient_id, {})
        missing = [name for name, is_required in required.items() if is_required and name not in modalities]
        case = CaseRecord(
            patient_id=patient_id,
            dataset=schema["name"],
            modalities=modalities,
            segmentation=seg_paths.get(patient_id),
            extra={"required": required},
        )
        cases.append(case)
        if missing:
            incomplete.append({"patient_id": patient_id, "missing_modalities": missing})

    unmatched = sorted(str(path.relative_to(root)) for path in nifti_files if path not in matched)
    if not cases:
        raise DatasetLayoutError(
            f"NIfTI files exist under {root} but none matched {schema['name']} patterns. "
            f"Unmatched examples: {unmatched[:8]}. Update the schema from inspect_dataset.py."
        )
    return DiscoveryReport(
        dataset=schema["name"],
        root=root,
        cases=cases,
        incomplete=incomplete,
        unmatched_files=unmatched,
    )


def require_complete(report: DiscoveryReport) -> list[CaseRecord]:
    if report.incomplete:
        preview = report.incomplete[:8]
        raise MissingModalityError(
            f"{report.dataset} has {len(report.incomplete)} incomplete patients: {preview}. "
            "Missing modalities are not ignored."
        )
    return report.cases


def list_dicom_series(root: Path) -> list[dict[str, Any]]:
    """List DICOM series. Reading headers does not modify the files."""
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise DatasetLayoutError("SimpleITK is required to read DICOM series.") from exc

    found: list[dict[str, Any]] = []
    folders = [root, *[path for path in root.rglob("*") if path.is_dir()]]
    for folder in folders:
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or ()
        except Exception:
            continue
        for series_id in series_ids:
            files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), series_id))
            if not files:
                continue
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(files)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.Execute()

            def meta(tag: str) -> str:
                if reader.HasMetaDataKey(0, tag):
                    return str(reader.GetMetaData(0, tag)).strip()
                return ""

            found.append(
                {
                    "patient_id": meta("0010|0020"),
                    "series_description": meta("0008|103e"),
                    "series_id": series_id,
                    "n_files": len(files),
                    "directory": str(folder),
                    "files": files,
                }
            )
    return found


def _discover_dicom(schema: dict[str, Any], root: Path) -> DiscoveryReport:
    mapping = {
        str(key).strip(): str(value).strip()
        for key, value in (schema.get("series_description_map") or {}).items()
    }
    series = list_dicom_series(root)
    if not series:
        raise DatasetLayoutError(f"No DICOM series under {root}.")
    if not mapping:
        descriptions = sorted({item["series_description"] or "<empty>" for item in series})
        raise DatasetLayoutError(
            f"{schema['name']} series_description_map is empty. "
            f"Observed descriptions: {descriptions}. "
            "Map those descriptions to t1, t1ce, t2, or flair. Do not guess."
        )
    known = set(schema.get("modalities", {}))
    for target in mapping.values():
        if target not in known:
            raise DatasetLayoutError(f"series_description_map target '{target}' is not a modality.")

    required = {
        name: bool(spec.get("required", False))
        for name, spec in schema.get("modalities", {}).items()
    }
    grouped: dict[str, dict[str, Path]] = {}
    sources: dict[str, dict[str, list[str]]] = {}
    matched_ids: set[str] = set()
    for item in series:
        modality = mapping.get(item["series_description"])
        if modality is None:
            continue
        patient_id = item["patient_id"] or Path(item["directory"]).name
        slot = grouped.setdefault(patient_id, {})
        if modality in slot:
            raise DatasetLayoutError(
                f"Multiple {modality} series for patient {patient_id}. "
                "The schema map is ambiguous."
            )
        slot[modality] = Path(item["files"][0])
        sources.setdefault(patient_id, {})[modality] = list(item["files"])
        matched_ids.add(item["series_id"])

    cases: list[CaseRecord] = []
    incomplete: list[dict[str, Any]] = []
    for patient_id in sorted(grouped):
        missing = [name for name, is_required in required.items() if is_required and name not in grouped[patient_id]]
        cases.append(
            CaseRecord(
                patient_id=patient_id,
                dataset=schema["name"],
                modalities=grouped[patient_id],
                extra={"required": required, "dicom_files": sources[patient_id]},
            )
        )
        if missing:
            incomplete.append({"patient_id": patient_id, "missing_modalities": missing})
    if not cases:
        raise DatasetLayoutError(
            f"DICOM series exist under {root} but none matched series_description_map."
        )
    unmatched = sorted(
        item["series_description"] or "<empty>"
        for item in series
        if item["series_id"] not in matched_ids
    )
    return DiscoveryReport(
        dataset=schema["name"],
        root=root,
        cases=cases,
        incomplete=incomplete,
        unmatched_files=unmatched,
    )
