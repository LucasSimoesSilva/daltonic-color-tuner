from cluster_cvd_step import *

centroids_lab, pal_cvd_lab, D_cvd, edges = analyze_collisions(
    "../images/ishiharaTest.png", cluster_number=6, cvd_type="deutan", severity=1.0, threshold=8.0
)
