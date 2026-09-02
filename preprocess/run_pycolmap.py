import argparse
import shutil
from pathlib import Path

import pycolmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-path", required=True)
    parser.add_argument("--single-camera", action="store_true")
    parser.add_argument("--max-image-size", type=int, default=1600)
    args = parser.parse_args()

    source = Path(args.source_path)
    images = source / "images"
    database = source / "database.db"
    sparse = source / "sparse"
    if database.exists():
        database.unlink()
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)

    reader_options = pycolmap.ImageReaderOptions()
    camera_mode = (
        pycolmap.CameraMode.SINGLE
        if args.single_camera
        else pycolmap.CameraMode.AUTO
    )
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.max_image_size = args.max_image_size

    print("Extracting SIFT features...")
    pycolmap.extract_features(
        database,
        images,
        camera_mode=camera_mode,
        reader_options=reader_options,
        extraction_options=extraction_options,
    )

    print("Matching all image pairs...")
    pycolmap.match_exhaustive(database)

    print("Running incremental mapping...")
    reconstructions = pycolmap.incremental_mapping(database, images, sparse)
    if not reconstructions:
        raise RuntimeError("COLMAP did not produce a reconstruction.")

    ranked = sorted(
        reconstructions.items(),
        key=lambda item: item[1].num_reg_images(),
        reverse=True,
    )
    best_id, best = ranked[0]
    best_dir = sparse / str(best_id)
    if best_dir != sparse / "0":
        target = sparse / "0"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(best_dir, target)

    registered = sorted(image.name for image in best.images.values())
    all_images = sorted(path.name for path in images.iterdir() if path.is_file())
    missing = sorted(set(all_images) - set(registered))
    (source / "colmap_registered.txt").write_text(
        "\n".join(registered) + "\n", encoding="utf-8"
    )
    (source / "colmap_missing.txt").write_text(
        "\n".join(missing) + ("\n" if missing else ""), encoding="utf-8"
    )

    print(f"Best model: {best_id}")
    print(f"Registered images: {len(registered)}/{len(all_images)}")
    print(f"3D points: {best.num_points3D()}")
    if missing:
        print("Unregistered images: " + ", ".join(missing))


if __name__ == "__main__":
    main()
