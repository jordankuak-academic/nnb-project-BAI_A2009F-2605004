import hashlib
import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from PIL import Image, ImageOps, UnidentifiedImageError
from configs.settings import CONFIGS


class Preprocessor:
    def __init__(self, root: Path | None = None) -> None:
        self.__config = CONFIGS["PREPROCESSOR"]
        self.__root = Path(root) if root is not None else (Path(__file__).resolve().parents[2] / "datasets" / "images")
        self.__unprocessed  = self.__root / "unprocessed"
        self.__processed = self.__root / "processed"
    
    def run(self) -> dict:
        backups = list(self.__root.glob(".preprocess-backup-*"))                 # Preserve recovery backups left by an interrupted or failed publication.
        
        if backups:
            raise RuntimeError(f"Resolve unfinished dataset backups before running: {backups}")
        
        existing_stats = self.__load_existing_stats()
        if existing_stats is not None:
            return existing_stats                                                      # Reuse the saved split without reading or processing raw images.
        
        seed = self.__validate_outputs_and_read_seed()                                 # Validate and read the random seed from split_seed.txt.
        files = self.__find_images()                                                   # Collect supported image paths in a deterministic order.
        
        # Stage results in a temporary directory that is cleaned up when this block exits.
        with TemporaryDirectory(prefix=".preprocess-", dir=self.__root) as temp:
            stage = Path(temp)                                                         # Convert the temporary directory name into a filesystem path.
            groups, duplicates, rejected = self.__scan_images(files, stage / "cache")  # Prepare and cache unique images; record skipped duplicates and unreadable images.
            classes, samples, counts = self.__split_images(groups, seed)               # Create class indices, reproducible per-class splits, and per-class counts.
            self.__export_images(stage, samples)                                       # Write numbered images to staged all/split directories and update sample paths.
            
            # Build split.json: preprocessing metadata and labeled sample locations.
            manifest = {
                **self.__output_config(seed),                                          # Save the same configuration used by reuse checks.
                "class_to_idx": classes,                                               # Map class names to integer target labels.
                "samples": samples,                                                    # Unique sample records with labels, sources, hashes, and output paths.
            }
            
            # Build split_stat.json: dataset sizes and data-quality audit records.
            stats = {
                "raw_candidates": len(files),                                          # Supported input files found before decoding or deduplication.
                "unique_images": len(samples),                                         # Retained unique samples, counted once despite two output copies.
                "duplicates_removed": len(duplicates),                                 # Duplicates skipped from output; raw files remain untouched.
                "label_conflicts": sum(d["label_conflict"] for d in duplicates),       # Count True flags for cross-class duplicates.
                "duplicates": duplicates,                                              # Kept/skipped paths, original labels, and matching pixel hashes.
                "rejected_count": len(rejected),                                       # Number of images skipped because image preparation failed.
                "rejected": rejected,                                                  # Source paths and error messages for rejected images.
                "per_class": counts,                                                   # Actual all/training/validating/testing counts for each class.
                "totals": {
                    subset: sum(count[subset] for count in counts.values())            # Sum this subset across all classes.
                    for subset in ("all", *self.__config["SPLIT_RATIOS"])              # Include all and unpack the split-name keys.
                },
            }
            self.__publish(stage, seed, manifest, stats)
        
        return stats
    
    def __output_config(self, seed: int) -> dict:
        return {
            "version": 1,                                                             # Manifest schema version, not the model version.
            "seed": seed,                                                             # Random seed used to reproduce the dataset split.
            "image_size": list(self.__config["IMAGE_SIZE"]),                     # Output [width, height] in pixels.
            "mode": "RGB",                                                            # Three-channel output color mode.
            "resize": "pad_black_lanczos",                                            # Aspect-preserving LANCZOS resize with black padding.
            "ratios": dict(self.__config["SPLIT_RATIOS"]),                  # Requested split proportions before integer rounding.
            "extensions": sorted(self.__config["EXTENSIONS"]),
            "duplicate_policy": "keep_first_sorted_source_path",                      # Retain the first pixel-identical image by path order.
        }
    
    def __load_existing_stats(self) -> dict | None:
        # Reuse the existing split only if every output image, metadata file and statistic is consistent.
        # Any mismatch (missing image, wrong hash, changed config, invalid class map) triggers a rebuild.
        manifest_path = self.__root / "split.json"
        stats_path = self.__root / "split_stat.json"
        actual_files = {p.relative_to(self.__root).as_posix() for p in self.__processed.rglob("*") if p.is_file()}
        has_metadata = any(p.exists() and p.stat().st_size for p in (manifest_path, stats_path))
        
        if not actual_files and not has_metadata:
            return None                                                               # Empty output directories and placeholder files are allowed.
        
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            stats = json.loads(stats_path.read_text(encoding="utf-8-sig"))
            seed = int((self.__root / "split_seed.txt").read_text(encoding="utf-8-sig").strip())
            expected_config = self.__output_config(seed)
            
            if not 0 <= seed < 2 ** 32 or any(manifest[key] != value for key, value in expected_config.items()):
                raise ValueError("Saved configuration or seed does not match.")
            
            classes = manifest["class_to_idx"]
            if not isinstance(classes, dict) or len(classes) < 2 or classes != {label: i for i, label in enumerate(sorted(classes))}:
                raise ValueError("Invalid class mapping.")
            
            samples = manifest["samples"]
            if not isinstance(samples, list) or not samples:
                raise ValueError("The sample manifest is empty or invalid.")
            
            # Recount per-class totals from the manifest and verify that each referenced file exists.
            counts = {label: dict(all=0, training=0, validating=0, testing=0) for label in classes}
            expected_files, identifiers, hashes = set(), set(), set()
            
            for sample in samples:
                identifier = sample["id"]
                
                if not isinstance(identifier, str) or not identifier.startswith("sample_image_") or not identifier.removeprefix("sample_image_").isascii() or not identifier.removeprefix("sample_image_").isdigit() or identifier in identifiers:
                    raise ValueError("Invalid or duplicate sample ID.")
                
                identifiers.add(identifier)
                digest = sample["pixel_sha256"]
                
                if digest in hashes:
                    raise ValueError("Duplicate sample hash in manifest.")
                
                hashes.add(digest)
                label, subset = sample["label"], sample["split"]
                
                if subset not in self.__config["SPLIT_RATIOS"] or sample["class_index"] != classes[label]:
                    raise ValueError("Invalid sample label or split.")
                
                counts[label]["all"] += 1
                counts[label][subset] += 1
                
                for field, folder in (("all_path", "all"), ("path", subset)):
                    relative = f"processed/{folder}/{identifier}.png"
                    
                    if sample[field] != relative:
                        raise ValueError(f"Unexpected sample path: {sample[field]}")
                    
                    path = self.__root / relative
                    
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError(f"Missing or empty output image: {relative}")
                    
                    expected_files.add(relative)
            
            totals = {subset: sum(c[subset] for c in counts.values()) for subset in ("all", *self.__config["SPLIT_RATIOS"])}
            
            if actual_files != expected_files or stats["unique_images"] != len(samples) or stats["per_class"] != counts or stats["totals"] != totals or any(c[s] < 1 for c in counts.values() for s in self.__config["SPLIT_RATIOS"]):
                raise ValueError("Output inventory or split statistics do not match the manifest.")
            
            return stats
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            logging.getLogger(__name__).warning("Rebuilding dataset: %s", error)
            return None                                                               # Leave old outputs untouched until new results are ready.
    
    def __validate_outputs_and_read_seed(self) -> int:
        if not self.__unprocessed.is_dir():
            raise FileNotFoundError(self.__unprocessed)
        if self.__processed.is_symlink() or (self.__processed.exists() and not self.__processed.is_dir()):
            raise ValueError(f"Expected a regular output directory: 'processed'")
        
        # Persist the seed in split_seed.txt so the split can be reproduced across runs.
        seed_path = self.__root / "split_seed.txt"
        contents  = seed_path.read_text(encoding="utf-8-sig").strip() if seed_path.exists() else ""
        
        try:
            seed = int(contents) if contents else self.__config["DEFAULT_SEED"]
        except ValueError as error:
            raise ValueError(f"Expected an integer in {seed_path}") from error
        if not 0 <= seed < 2 ** 32:
            raise ValueError("Seed must be in [0, 2**32).")
        
        return seed
    
    def __find_images(self) -> list[Path]:
        # Sort by relative path so the input order is deterministic regardless of filesystem traversal order.
        return sorted((p for p in self.__unprocessed.rglob("*") if p.is_file() and p.suffix.lower() in self.__config["EXTENSIONS"]), key=lambda p: p.relative_to(self.__unprocessed).as_posix())
    
    def __prepare_image(self, path: Path) -> Image.Image:
        # 1. ImageOps.exif_transpose rotates the image according to its EXIF orientation tag, so photos taken in portrait/landscape are not stored sideways.
        # 2. convert("RGB") drops alpha channels and guarantees three channels for the CNN.
        # 3. ImageOps.pad resizes with aspect ratio preserved and pads the shorter side with black, avoiding distortion that would otherwise confuse the classifier.
        # 4. LANCZOS is a high-quality downsampling filter that reduces aliasing.
        with Image.open(path) as original:
            return ImageOps.pad(
                ImageOps.exif_transpose(original).convert("RGB"),
                self.__config["IMAGE_SIZE"],
                method=Image.Resampling.LANCZOS,
                color=(0, 0, 0),
            )
    
    def __scan_images(self, files: list[Path], cache: Path) -> tuple[dict, list, list]:
        # Pixel-level deduplication: each prepared image is hashed by its raw bytes (after resize/pad), 
        # so two files that look identical are detected even if their file names, formats or EXIF metadata differ. 
        # The first one in sorted path order is kept; later copies are recorded as duplicates and skipped.
        # This prevents the same image from leaking across train/val/test splits.
        groups = defaultdict(list)
        seen = {}
        duplicates = []
        rejected = []
        cache.mkdir()
        
        for path in files:
            source = path.relative_to(self.__root).as_posix()
            parts = path.relative_to(self.__unprocessed).parts
            
            if len(parts) != 3:
                raise ValueError(f"Expected unprocessed/source/class/image: {source}")
            
            label = parts[1]
            groups.setdefault(label, [])
            
            try:
                image = self.__prepare_image(path)
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                # Unreadable or suspiciously large images are skipped and logged, not fatal.
                rejected.append({"source": source, "error": str(error)})
                continue
            with image:
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                
                if digest in seen:
                    kept = seen[digest]
                    duplicates.append({
                        "pixel_sha256": digest,
                        "kept_source": kept["sources"][0],
                        "kept_label": kept["label"],
                        "skipped_source": source,
                        "skipped_label": label,
                        "label_conflict": kept["label"] != label,
                    })
                    kept["sources"].append(source)
                    continue
                
                # Save the prepared image once to the cache; __export_images copies it into
                # both processed/all and the split folder without re-decoding the raw file.
                image.save(cache / f"{digest}.png")
                record = {
                    "label": label,
                    "pixel_sha256": digest,
                    "sources": [source]
                }
                seen[digest] = record
                groups[label].append(record)
        
        return dict(groups), duplicates, rejected
    
    def __split_images(self, groups: dict, seed: int) -> tuple[dict, list, dict]:
        # Each class is shuffled independently with the same seed, then split into
        # training / validating / testing. Every split receives at least one image,
        # even for very small classes, so the DataLoader never sees an empty split.
        ratios = self.__config["SPLIT_RATIOS"]
        
        if set(ratios) != {"training", "validating", "testing"} or not all(0 < value < 1 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
            raise ValueError("Training/validating/testing ratios must be positive and sum to 1.")
        
        too_small = {label: len(items) for label, items in groups.items() if len(items) < 3}
        
        if len(groups) < 2 or too_small:
            raise ValueError(
                f"Need >=2 classes and >= 3 unique valid images per class; "
                f"class_count={len(groups)}, insufficient={too_small}"
            )
        
        rng = random.Random(seed)
        classes = {label: index for index, label in enumerate(sorted(groups))}
        samples = []
        counts = {}
        
        for label, class_index in classes.items():
            items = list(groups[label])
            rng.shuffle(items)
            total = len(items)
            n_validating = max(1, int(total * ratios["validating"]))
            n_testing = max(1, int(total * ratios["testing"]))
            n_training = total - n_validating - n_testing
            sizes = {
                "training": n_training,
                "validating": n_validating,
                "testing": n_testing
            }
            counts[label] = {
                "all": total,
                **sizes
            }
            start = 0
            
            for subset, size in sizes.items():
                for record in items[start:start + size]:
                    samples.append({
                        **record,
                        "split": subset,
                        "class_index": class_index
                    })
                start += size
        
        rng.shuffle(samples)
        
        return classes, samples, counts
    
    @staticmethod
    def __export_images(stage: Path, samples: list[dict]) -> None:
        # Each unique image is written twice: once into processed/all (for inspection)
        # and once into its split folder (training/validating/testing) for the DataLoader.
        # The two copies share the same file name, so a sample's identity is preserved.
        for index, record in enumerate(samples, 1):
            name = f"sample_image_{index:04d}.png"
            record["id"] = Path(name).stem
            cached_image = stage / "cache" / f"{record['pixel_sha256']}.png"
            destinations = {
                "all_path": "all",
                "path": record["split"]
            }
            
            for field, subset in destinations.items():
                relative_path = Path("processed") / subset / name
                target = stage / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached_image, target)
                record[field] = relative_path.as_posix()
    
    def __publish(self, stage: Path, seed: int, manifest: dict, stats: dict) -> None:
        # Atomic publication: move old artefacts aside, install staged ones, then
        # delete backups. If installation fails at any point, restore the old dataset
        # so a crash never leaves the user with a half-written processed/ directory.
        for name, data in (("split.json", manifest), ("split_stat.json", stats)):
            content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
            (stage / name).write_text(content, encoding="utf-8")
        
        (stage / "split_seed.txt").write_text(f"{seed}\n", encoding="utf-8")
        names = ("split_seed.txt", "split_stat.json", "split.json")
        
        for name in names:
            target = self.__root / name
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError(f"Expected a regular metadata file: {target}")
        
        # Backups live outside the stage directory so TemporaryDirectory cleanup cannot destroy them.
        backup = Path(mkdtemp(prefix=".preprocess-backup-", dir=self.__root))
        replacements = [(
            stage / "processed",
            self.__processed,
            backup / "processed"
        )]
        replacements.extend(
            (stage / name, self.__root / name, backup / name) for name in names
        )
        saved = []
        installed = []
        
        try:
            # Move every old artifact aside before installing any new artifact.
            for source, target, old in replacements:
                if target.exists():
                    target.replace(old)
                    saved.append((old, target))
            for source, target, old in replacements:
                # Replace the directory as a whole, never merge its contents with the old one.
                source.replace(target)
                installed.append(target)
        except BaseException:
            # Roll back any installed artefacts and restore the previous ones.
            try:
                for target in reversed(installed):
                    if target.is_dir():
                        shutil.rmtree(target)
                    elif target.exists():
                        target.unlink()
                for old, target in reversed(saved):
                    old.replace(target)
            except BaseException as recovery_error:
                raise RuntimeError(
                    f"Dataset rollback failed. Preserve and recover files from {backup}."
                ) from recovery_error
            else:
                shutil.rmtree(backup)
            raise
        else:
            # Installation succeeded; the previous dataset is now obsolete.
            try:
                shutil.rmtree(backup)
            except OSError as error:
                raise RuntimeError(
                    f"New dataset installed, but obsolete backup cleanup failed: {backup}. "
                    "Verify the new outputs and remove this backup before running again."
                ) from error


if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.run()
