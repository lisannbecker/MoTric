import argparse
from trainer import train_prior as led


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
	t = led.Trainer(config)
	if config.train==1:
		t.fit()
	elif config.train==0:
		# t.save_data()
		t.test_single_model()
	elif config.train==2:
		t.simulate_algorithm_and_correct_synthetic()
	elif config.train==3:
		t.simulate_algorithm_and_correct_clusters_for_bandw()

if __name__ == "__main__":
	config = parse_config()
	main(config)
