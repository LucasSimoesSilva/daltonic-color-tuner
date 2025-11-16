from color_utils import *
from cluster_utils import *
from color_utils import lab_to_rgb, write_rgb


def kmeans_ab(ab_samples: np.ndarray, cluster_number: int = 6, attempts: int = 5, criteria_eps: float = 0.5,
              criteria_iter: int = 50):
    """
    k-means using OpenCV on a*b*(float32).
    Returns centroids (k,2) and labels (n,).
    """

    # Points in the format (n,2)
    points = ab_samples.reshape(-1, 2)

    # Stopping criterion: when the maximum number of iterations is reached or when there is no significant improvement.
    # (criteria_type, max_iter, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, criteria_iter, criteria_eps)

    compactness, labels, cluster_centers = cv2.kmeans(
        points, K=cluster_number, bestLabels=None, criteria=criteria, attempts=attempts, flags=cv2.KMEANS_PP_CENTERS)

    # Flatten labels to a 1D vector
    labels = labels.reshape(-1)

    # Return cluster centers and labels
    return cluster_centers.astype(np.float32), labels


def simulate_palette_cvd(centroids_lab: np.ndarray, cvd_type: str, severity: float) -> np.ndarray:
    """
    Converts palette Lab -> RGB -> simulates CVD -> converts back to Lab.
    centroids_lab: (k,3)
    Returns: (k,3) in Lab already simulated.
    """

    # shape: (k,3)
    palette_rgb = lab_to_rgb(centroids_lab)

    # Treat the palette as a 1xk "image" because simulate_cvd expects (H, W, 3)
    palette_rgb_img = palette_rgb[None, :, :]

    palette_cvd_rgb_img = simulate_cvd(palette_rgb_img, cvd_type=cvd_type, severity=severity)

    # Back to shape (k,3)
    palette_cvd_rgb = palette_cvd_rgb_img.reshape(-1, 3)

    palette_cvd_lab = color.rgb2lab(palette_cvd_rgb).astype(np.float32)
    return palette_cvd_lab


def pairwise_delta_lab(palette_lab: np.ndarray) -> np.ndarray:
    """
    Computes a ΔE2000 distance matrix between two colors from the palette_lab.
    """
    num_colors = palette_lab.shape[0]
    delta_matrix = np.zeros((num_colors, num_colors), dtype=np.float32)

    for i in range(num_colors):
        # Repeat the i-th Lab color 'num_colors' times
        color_i_repeated = np.repeat(palette_lab[i][None, :], num_colors, axis=0)

        # Compute ΔE2000 between color i and every other color
        delta_matrix[i] = deltaE_2000(color_i_repeated, palette_lab)

    return delta_matrix


def collision_graph(delta_cvd_matrix: np.ndarray, threshold: float = 8.0):
    """
    Returns list of (i, j, ΔE) pairs where ΔE under CVD < threshold.
    These represent collisions (colors too similar after simulation).
    """
    num_colors = delta_cvd_matrix.shape[0]
    collision_edges = []

    for i in range(num_colors):
        for j in range(i + 1, num_colors):
            # If two colors become perceptually too similar, record this collision
            if delta_cvd_matrix[i, j] < threshold:
                collision_edges.append((i, j, float(delta_cvd_matrix[i, j])))

    return collision_edges


def analyze_collisions(image_path: str, cluster_number: int = 6, cvd_type: str = "deutan", severity: float = 1.0,
                       threshold: float = 8.0, output_path: str | None = "../images/saida_recolorida.png"):
    """
    Returns: (lab_centroids, lab_centroids_cvd, deltaE_matrix_cvd, collisions)
    """

    # 1) Read image and convert to Lab
    img_rgb = read_rgb(image_path)
    img_lab = rgb_to_lab(img_rgb)

    # 2) Sample a*b* points and run k-means on chroma only
    ab_samples, sample_rows, sample_cols = sample_ab_from_lab(img_lab, max_samples=50_000)
    ab_centers, cluster_labels = kmeans_ab(ab_samples, cluster_number=cluster_number)

    # 3) Build full Lab centroids
    lab_centroids = build_lab_centroids(img_lab, cluster_labels, sample_rows, sample_cols, ab_centers)

    # 4) Simulate CVD on centroids and compute pairwise ΔE between simulated centroids
    lab_centroids_cvd = simulate_palette_cvd(lab_centroids, cvd_type=cvd_type, severity=severity)
    delta_cvd = pairwise_delta_lab(lab_centroids_cvd)

    # 5) Collisions (ΔE < threshold)
    collisions = collision_graph(delta_cvd, threshold=threshold)

    # 6) Print
    print(f"Clusters: {cluster_number}, CVD: {cvd_type}, severity: {severity}, threshold: {threshold}")
    print("ΔE2000 matrix under CVD:")
    np.set_printoptions(precision=2, suppress=True)
    print(delta_cvd)

    print_collisions(collisions)

    centroids_optimized = optimize_palette_from_collisions(
        lab_centroids,
        collisions,
        cvd_type=cvd_type,
        severity=severity,
        step=5.0,
        search_radius=2,
        lambda_fidelity=0.7,
    )

    palette_cvd_lab_optimized = simulate_palette_cvd(centroids_optimized, cvd_type=cvd_type, severity=severity)
    delta_cvd_optimized = pairwise_delta_lab(palette_cvd_lab_optimized)
    collisions_optimized = collision_graph(delta_cvd_optimized, threshold=threshold)

    print("After optimization:")
    print("\nΔE2000 matrix under CVD:")
    print(delta_cvd_optimized)

    print_collisions(collisions_optimized)

    # 7) Apply optimized palette to original image.
    print("\nGenerating recolored image")

    label_map_full = assign_clusters_to_image(img_lab, centroids_optimized)

    img_lab_recolored = recolor_image_from_clusters(
        img_lab=img_lab,
        label_map=label_map_full,
        lab_centroids_original=lab_centroids,
        lab_centroids_optimized=centroids_optimized,
        blend_ratio=0.1)

    img_rgb_recolored = lab_to_rgb(img_lab_recolored)
    write_rgb(output_path, img_rgb_recolored)

    print("Image created")

    return (lab_centroids, lab_centroids_cvd, delta_cvd, collisions, centroids_optimized,
            palette_cvd_lab_optimized, delta_cvd_optimized, collisions_optimized)


# lambda_fidelity -> Penalty weight for color change compared to the original color.
def optimize_single_centroid(
        cluster_index: int,
        palette_lab: np.ndarray,
        cvd_type: str = "deutan",
        severity: float = 1.0,
        step: float = 5.0,
        search_radius: int = 2,
        lambda_fidelity: float = 0.5) -> np.ndarray:
    """
    Optimizes a centroid (cluster_index) by moving only a* and b* in a grid.
    Objective: increase separation under CVD without significantly distorting the original color.
    Returns a new Lab centroid with shape(3,).
    """
    num_clusters = palette_lab.shape[0]
    original_centroid = palette_lab[cluster_index].copy()

    # In the start the best centroid is the original one
    best_centroid = original_centroid.copy()

    # less infinity for any initial score to be better
    best_score = -np.inf

    luminosity_original = float(original_centroid[0])
    a_original, b_original = float(original_centroid[1]), float(original_centroid[2])

    # Create an offset range that will be used for testing.
    # ex step 5 and search_radius 2: [-10, -5, 0, 5, 10]
    ab_offsets = np.arange(-search_radius, search_radius + 1, 1) * step

    for delta_a in ab_offsets:
        for delta_b in ab_offsets:
            # Applies the offset to the grid
            a_candidate = a_original + delta_a
            b_candidate = b_original + delta_b

            # New candidate centroid with a* and b* shifted
            candidate_centroid = np.array([luminosity_original, a_candidate, b_candidate], dtype=np.float32)

            # assemble a candidate palette
            candidate_palette = palette_lab.copy()
            candidate_palette[cluster_index] = candidate_centroid

            palette_cvd_lab = simulate_palette_cvd(candidate_palette, cvd_type=cvd_type, severity=severity)

            # Cluster color in CVD
            # (1,3)
            target_cvd_color = palette_cvd_lab[cluster_index][None, :]

            # Pallet in CVD
            # (k,3)
            all_cvd_colors = palette_cvd_lab

            # Calculate delta between target color and all colors in the CVD palette.
            delta_cvd = deltaE_2000(
                # Duplicates target_cvd_color k times to create pair (target, other colors).
                np.repeat(target_cvd_color, num_clusters, axis=0), all_cvd_colors)

            # Since the CVD delta needs to be minimal,
            # we define the distance between clusters of the same color as infinite so that it is not taken into account.
            delta_cvd[cluster_index] = np.inf
            min_delta_cvd = float(np.min(delta_cvd))

            # Delta between the candidate and the original color
            delta_normal = float(deltaE_2000(candidate_centroid[None, :], original_centroid[None, :])[0])

            # Objective function.
            # min_delta_cvd -> Increase the value to find more separated colors under CVD.
            # delta_normal -> Decrease the value of deltaE_normal to avoid altering the original color too much.
            # lambda_fidelity -> Penalty weight for color change compared to the original color.
            score = min_delta_cvd - lambda_fidelity * delta_normal

            # Check if the candidate is better than the previous one.
            if score > best_score:
                best_score = score
                best_centroid = candidate_centroid

    return best_centroid


def optimize_palette_from_collisions(
        palette_lab: np.ndarray,
        collision_edges,
        cvd_type: str = "deutan",
        severity: float = 1.0,
        step: float = 5.0,
        search_radius: int = 2,
        lambda_fidelity: float = 0.5) -> np.ndarray:
    """
    Optimizes only the clusters that participate in collisions.
    collision_edges: list of tuples (i, j, ΔE) from collision_graph.
    Returns new lab palette with shape (k,3).
    """

    if not collision_edges:
        return palette_lab.copy()

    # Creates a set with all cluster indices that appear on any edge.
    # This avoids duplicating optimizations of the same cluster.
    indices_to_optimize = set()
    for i, j, _ in collision_edges:
        indices_to_optimize.add(i)
        indices_to_optimize.add(j)

    # Starting optimized centroid variable
    optimized_centroids = palette_lab.copy()

    for cluster_index in sorted(indices_to_optimize):
        optimized_centroids[cluster_index] = optimize_single_centroid(
            cluster_index,
            optimized_centroids,
            cvd_type=cvd_type,
            severity=severity,
            step=step,
            search_radius=search_radius,
            lambda_fidelity=lambda_fidelity
        )

    return optimized_centroids


def recolor_image_from_clusters(
        img_lab: np.ndarray,
        label_map: np.ndarray,
        lab_centroids_original: np.ndarray,
        lab_centroids_optimized: np.ndarray,
        blend_ratio: float = 0.1) -> np.ndarray:
    """
    Recolors the image by replacing each cluster with its optimized centroid.
    blend_ratio: fraction of the original color blended to avoid harsh edges. (0 = optimized only; 1 = original only).
    """
    height, width, _ = img_lab.shape

    # Starting image as a zero array
    img_lab_recolored = np.zeros_like(img_lab, dtype=np.float32)

    # Get number of clusters based on number of centroids
    num_clusters = lab_centroids_original.shape[0]

    for cluster_index in range(num_clusters):
        # Get the pixels that are in this cluster
        cluster_mask = (label_map == cluster_index)

        # Check if cluster is really in the image
        if np.any(cluster_mask):
            # Calculates the final color with blend:
            # Greater weighting on the optimized centroid (1.0 - blend_ratio);
            # Small fraction of the original centroid to smooth transitions (blend_ratio);
            blended_centroid = (
                    (1.0 - blend_ratio) * lab_centroids_optimized[cluster_index] +
                    blend_ratio * lab_centroids_original[cluster_index])

            # Assigns this mixed color to all pixels belonging to this cluster.
            img_lab_recolored[cluster_mask] = blended_centroid

    return img_lab_recolored


def assign_clusters_to_image(img_lab: np.ndarray, centroids_lab: np.ndarray) -> np.ndarray:
    """
    Assigns a cluster label (0..k-1) to each pixel in the image,
    based on the distance in a*b* to the centroids.
    Returns an array (H, W) of integer labels.
    """
    height, width, _ = img_lab.shape

    # extracts only the a* and b* channels from the image.
    # (N,2)
    ab_pixels = img_lab[..., 1:3].reshape(-1, 2).astype(np.float32)

    # extracts the a* and b* channels from the centroids (ignores Luminosity)
    # (k,2)
    ab_centroids = centroids_lab[:, 1:3].astype(np.float32)

    # Calculates difference between each pixel and each centroid.
    diff = ab_pixels[:, None, :] - ab_centroids[None, :, :]

    # Euclidean distance squared between a pixel and each centroid
    # (N,k)
    dist_squared = np.sum(diff * diff, axis=2)

    # For each pixel, it retrieves the index of the nearest centroid.
    cluster_labels = np.argmin(dist_squared, axis=1).reshape(height, width)
    return cluster_labels

def print_collisions(collisions: list[tuple[int, int, float]]) -> None:
    if collisions:
        print("\nCollisions detected (i, j, ΔE):")
        for i, j, d in collisions:
            print(f"  {i} -- {j}  ΔE={d:.2f}")
    else:
        print("\nNo collisions below the threshold.")