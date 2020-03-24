#ifndef LOAD_MNIST_DATASET_H
#define LOAD_MNIST_DATASET_H

#include <fstream>
#include <vector>

struct MNISTPair {
	std::vector<uint8_t> image;
	uint8_t label;
};

struct MNISTDataset {
	uint32_t num_items;
	uint32_t img_width;
	uint32_t img_height;
	std::vector<MNISTPair> data;
};

MNISTDataset load_mnist_dataset(const char* labels_file, const char* images_file);

#endif // !LOAD_MNIST_DATASET_H