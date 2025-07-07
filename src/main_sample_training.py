from model_controller import ModelTrainController


def simple_train():
    lens = [16]
    levels = [
        '../dataset/handwritten-only-8lines--2388.zip',
    ]
    train_name = "simple-8lines-training"

    model = ModelTrainController(NUM_LINES=8, NO_TEACH=False)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       levels, 0.25, 0.90,
                                       (1, 1),   # 3 epochs, para teste somente...
                                       (1000, 200),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINISHED TEST => ", train_name)


if __name__ == '__main__':
    simple_train()
