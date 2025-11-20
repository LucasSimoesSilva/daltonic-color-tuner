import os
import glob
import numpy as np

from cluster_cvd_step import analyze_collisions


def evaluate_dataset(
        input_folder: str, cluster_number: int = 6,
        cvd_type: str = "deutan", severity: float = 1.0,
        collision_threshold: float = 8.0,
        output_folder: str = "resultados_ishihara", apply_luminosity: bool = False):
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
            output_folder, image_name + "_recolored.png"
        )

        print("\n=== Processing:", image_path, "===")

        (lab_centroids, palette_cvd_lab, delta_cvd, collisions, lab_centroids_optimized,
         palette_cvd_lab_optimized, delta_cvd_optimized, collisions_optimized) = analyze_collisions(
            image_path, cluster_number=cluster_number, cvd_type=cvd_type, severity=severity,
            threshold=collision_threshold, output_path=recolored_output_path, apply_luminosity=apply_luminosity)

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


def evaluate_images():
    real_num = {
        "image_1_recolored.png": "5",
        "image_2_recolored.png": "3",
        "image_3_recolored.png": "15",
        "image_4_recolored.png": "74",
        "image_5_recolored.png": "6",
        "image_6_recolored.png": "45",
        "image_7_recolored.png": "5",
        "image_8_recolored.png": "7",
        "image_9_recolored.png": "16",
        "image_10_recolored.png": "73",
        "image_11_recolored.png": "26",
        "image_12_recolored.png": "42",
        "image_13_recolored.png": "12",
        "image_14_recolored.png": "8",
        "image_15_recolored.png": "29",
    }

    import os
    import random
    from PIL import Image

    folder = "./resultados_ishihara"

    files = [
        arq for arq in os.listdir(folder)
    ]

    random.shuffle(files)

    for name in files:
        image_path = os.path.join(folder, name)

        img = Image.open(image_path)
        img.show()

        right_num = real_num.get(name)

        answer = input(f"What number do you see in the image {name}? ")

        if answer.strip() == right_num:
            print("✅ Correct!\n")
        else:
            print(f"❌ You're wrong. The correct number was: {right_num}\n")

        img.close()


if __name__ == "__main__":
    option = int(input('1 - If you want to recolonized the images\n'
                       '2 - If you want to evaluate the images\n'
                       'Option:'))
    if option == 1:
        evaluate_dataset(
            input_folder="../images/ishiharaPlates",
            cluster_number=12,
            cvd_type="deutan",
            severity=0.8,
            collision_threshold=8.0,
            output_folder="resultados_ishihara", apply_luminosity=False
        )
    elif option == 2:
        evaluate_images()
