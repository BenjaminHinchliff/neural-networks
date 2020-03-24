#include "load_mnist_dataset.h"

// helper function to extract numbers from the start of the mnist datasets
uint32_t four_stream_bytes_to_int(std::ifstream& stream) {
	uint32_t output = 0u;
	for (int i = 24; i >= 0; i -= 8) {
		uint8_t byte;
		stream >> byte;
		output |= (static_cast<uint32_t>(byte) << i);
	}
	return output;
}

MNISTDataset load_mnist_dataset(const char* labels_file, const char* images_file) {
	std::ifstream labels_file_handle(labels_file, std::ios::binary);
	labels_file_handle.ignore(8); // ignore magic numbers and useless data
	std::ifstream images_file_handle(images_file, std::ios::binary);
	images_file_handle.ignore(4); // ignore magic numbers
	MNISTDataset dataset{
		four_stream_bytes_to_int(images_file_handle),
		four_stream_bytes_to_int(images_file_handle),
		four_stream_bytes_to_int(images_file_handle),
		{}
	};

	uint32_t img_size = dataset.img_width * dataset.img_height;
	std::vector<uint8_t> img;
	img.resize(img_size);
	for (uint32_t i = 0u; i < dataset.num_items; ++i) {
		images_file_handle.read(reinterpret_cast<char*>(&img[0]), img_size);
		uint8_t label;
		labels_file_handle.read(reinterpret_cast<char*>(&label), 1);
		MNISTPair sample_pair{ img, label };
		dataset.data.push_back(sample_pair);
	}
	return dataset;
}