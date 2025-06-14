import argparse
import pickle
import numpy as np
from collections import defaultdict

from trainer.train_prior import Trainer, OfflineTrajectoryClusterer

# grid of candidates
H_GRID      = [0.2, 0.4, 0.6, 0.8, 1.0]
LAMBDA_GRID = [0.1, 0.2, 0.4, 0.8]

def parse_config():
	parser = argparse.ArgumentParser()
	parser.add_argument("--cuda", default=True)
	parser.add_argument("--learning_rate", type=int, default=0.002)
	parser.add_argument("--max_epochs", type=int, default=128)

	parser.add_argument('--cfg', default='led_augment')
	parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use.')
	parser.add_argument('--train', type=int, default=1, help='Whether train or evaluate.')
	
	parser.add_argument("--info", type=str, default='', help='Name of the experiment. '
															 'It will be used in file creation.')
	parser.add_argument("--dataset", type=str, default='kitti', help='Name of the dataset (nba / kitti / newer/ spires).') #overrides config file
	return parser.parse_args()


def main(config):
    # 1) Initialize trainer & data
    trainer = Trainer(config)

    # 2) Build clusters
    clusterer = OfflineTrajectoryClusterer(
        trainer,
        num_clusters=5,
        pca_dim=20,
        save_path="trajectory_cluster_model.pkl"
    )
    clusterer.fit()
	
    # 2) Assign every training trajectory to a cluster
    all_feats  = clusterer.gather_features(trainer.train_loader)  # shape (N_total, T_past*7)
    all_labels = clusterer.kmeans.predict(clusterer.pca.transform(all_feats))

    #group indices by cluster
    from collections import defaultdict
    cluster_indices = defaultdict(list)
    for idx, lab in enumerate(all_labels):
        cluster_indices[lab].append(idx)

    #do grid search
    # candidate grids
    H_GRID      = [0.2, 0.4, 0.6, 0.8, 1.0]
    LAMBDA_GRID = [0.1, 0.2, 0.4, 0.8]

    cluster_to_h = {}
    cluster_to_lambda = {}

    for j, indices in cluster_indices.items():
        # split into inner-train / inner-val
        split = int(0.8 * len(indices))
        train_idx, val_idx = indices[:split], indices[split:]

        best_ade = float('inf')
        best_pair = (None, None)

        for h_c in H_GRID:
            for λ_d in LAMBDA_GRID:
                # 1) override your Trainer to use (h_c, λ_d)
                trainer.override_bandwidth_and_lambda(h=h_c, λ=λ_d)

                # 2) evaluate on the *validation subset* of cluster j
                #    this runs exactly the same correction loop, but only on those val_idx
                ade_val = trainer.evaluate_correction_on_indices(val_idx)

                # 3) keep the best
                if ade_val < best_ade:
                    best_ade = ade_val
                    best_pair = (h_c, λ_d)

        cluster_to_h[j], cluster_to_lambda[j] = best_pair
        print(f"Cluster {j}: best h={best_pair[0]}, λ={best_pair[1]} (ADE={best_ade:.4f})")


if __name__ == "__main__":
	config = parse_config()
	main(config)
