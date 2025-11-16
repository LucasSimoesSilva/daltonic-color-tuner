import os
import glob
import numpy as np

from cluster_cvd_step import analyze_collisions


def evaluate_dataset(
        input_folder: str, cluster_number: int = 6,
        cvd_type: str = "deutan", severity: float = 1.0,
        collision_threshold: float = 8.0,
        output_folder: str = "resultados_ishihara"):
    """
    Evaluate all images in a folder by:
    - running the collision analysis and palette optimization;
    - computing before/after CVD distances and number of collisions;
    """

    os.makedirs(output_folder, exist_ok=True)

    # Collect all images paths in the input folder
    image_paths = sorted(
        glob.glob(os.path.join(input_folder, "*.png"))
        + glob.glob(os.path.join(input_folder, "*.jpg"))
        + glob.glob(os.path.join(input_folder, "*.jpeg"))
    )

    if not image_paths:
        print("No images in the folder:", input_folder)
        return

    image_summaries = []

    for image_path in image_paths:
        # Base file name without extension (used to build output paths)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        recolored_output_path = os.path.join(
            output_folder, image_name + "_recolorida.png"
        )

        print("\n=== Processing:", image_path, "===")

        (lab_centroids, palette_cvd_lab, delta_cvd, collisions, lab_centroids_optimized,
         palette_cvd_lab_optimized, delta_cvd_optimized, collisions_optimized) = analyze_collisions(
            image_path, cluster_number=cluster_number, cvd_type=cvd_type, severity=severity,
            threshold=collision_threshold, output_path=recolored_output_path)

        # --- Metrics per image ---
        # Ignore diagonal (self-distance = 0) for min ΔE
        num_clusters = delta_cvd.shape[0]
        off_diagonal_mask = ~np.eye(num_clusters, dtype=bool)

        # Minimum pairwise distance between different clusters (before optimization)
        min_delta_before = float(delta_cvd[off_diagonal_mask].min())

        # Minimum pairwise distance between different clusters (after optimization)
        min_delta_after = float(delta_cvd_optimized[off_diagonal_mask].min())

        # Number of collisions
        num_collisions_before = len(collisions)
        num_collisions_after = len(collisions_optimized)

        image_summaries.append((image_name, min_delta_before, min_delta_after,
                                num_collisions_before, num_collisions_after))

    # Print a global summary for all images
    print("\n=== Summary per image ===")
    print("Image, min_DE_before, min_DE_after, collisions_before, collisions_after")
    for row in image_summaries:
        print("{}, {:.2f}, {:.2f}, {}, {}".format(*row))


if __name__ == "__main__":
    # TODO: Add dataset
    evaluate_dataset(
        input_folder="data/ishihara",
        cluster_number=6,
        cvd_type="deutan",
        severity=1.0,
        collision_threshold=8.0,
        output_folder="resultados_ishihara",
    )
