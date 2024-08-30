from .dataloader_nyu import NyuLoader

class Args:
    def __init__(self):
        self.input_height = 480
        self.input_width = 640
        self.batch_size = 1
        self.num_threads = 4
        self.distributed = False
        self.data_augmentation_hflip = True
        self.data_augmentation_color = True
        self.data_augmentation_random_crop = False

if __name__ == '__main__':
    args = Args()
    train_loader = NyuLoader(args, 'train').data
    test_loader = NyuLoader(args, 'test').data

    train_data = train_loader[0]
    test_data = test_loader[0]

    print(train_data)
    print(test_data)