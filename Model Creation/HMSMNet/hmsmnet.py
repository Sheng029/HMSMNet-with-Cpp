import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow.keras as keras
from feature import FeatureExtraction
from cost import CostConcatenation
from aggregation import Hourglass, FeatureFusion
from computation import Estimation
from refinement import Refinement
from reader import read_left, read_right


class HMSMNet:
    def __init__(self, height, width, channel, min_disp, max_disp):
        self.height = height
        self.width = width
        self.channel = channel
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.model = None

    def build_model(self):
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))
        gx = keras.Input(shape=(self.height, self.width, self.channel))
        gy = keras.Input(shape=(self.height, self.width, self.channel))

        feature_extraction = FeatureExtraction(filters=16)
        [l0, l1, l2] = feature_extraction(left_image)
        [r0, r1, r2] = feature_extraction(right_image)

        cost0 = CostConcatenation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        cost1 = CostConcatenation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        cost2 = CostConcatenation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        cost_volume0 = cost0([l0, r0])
        cost_volume1 = cost1([l1, r1])
        cost_volume2 = cost2([l2, r2])

        hourglass0 = Hourglass(filters=16)
        hourglass1 = Hourglass(filters=16)
        hourglass2 = Hourglass(filters=16)
        agg_cost0 = hourglass0(cost_volume0)
        agg_cost1 = hourglass1(cost_volume1)
        agg_cost2 = hourglass2(cost_volume2)

        estimator2 = Estimation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        disparity2 = estimator2(agg_cost2)

        fusion1 = FeatureFusion(units=16)
        fusion_cost1 = fusion1([agg_cost2, agg_cost1])
        hourglass3 = Hourglass(filters=16)
        agg_fusion_cost1 = hourglass3(fusion_cost1)

        estimator1 = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        disparity1 = estimator1(agg_fusion_cost1)

        fusion2 = FeatureFusion(units=16)
        fusion_cost2 = fusion2([agg_fusion_cost1, agg_cost0])
        hourglass4 = Hourglass(filters=16)
        agg_fusion_cost2 = hourglass4(fusion_cost2)

        estimator0 = Estimation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        disparity0 = estimator0(agg_fusion_cost2)

        # refinement
        refiner = Refinement(filters=32)
        final_disp = refiner([disparity0, left_image, gx, gy])

        # self.model = keras.Model(inputs=[left_image, right_image, gx, gy],
        #                          outputs=[disparity2, disparity1, disparity0, final_disp])
        self.model = keras.Model(inputs=[left_image, right_image, gx, gy], outputs=final_disp)
        self.model.summary()


if __name__ == '__main__':
    #
    net = HMSMNet(1024, 1024, 1, -128.0, 64.0)
    net.build_model()
    net.model.load_weights('HMSM-Net.h5', by_name=True)
    print(net.model.input, net.model.output)
    net.model.save(filepath='hmsmnet', include_optimizer=False)
    #
    # #
    # model = keras.models.load_model('hmsmnet', compile=False)
    # model.summary()
    # print(model.input, model.output)
    #
    # #
    # left, dx, dy = read_left('../data/HY_left_0.tiff')
    # right = read_right('../data/HY_right_0.tiff')
    # left = np.expand_dims(left, 0)
    # dx = np.expand_dims(dx, 0)
    # dy = np.expand_dims(dy, 0)
    # right = np.expand_dims(right, 0)
    # disp = model.predict([left, right, dx, dy])[0, :, :, 0]
    # disp = Image.fromarray(disp)
    # disp.save('../data/pred.tiff')

    # #
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.imread('../data/HY_disparity_0.tiff', cv2.IMREAD_UNCHANGED))
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.imread('../data/pred.tiff', cv2.IMREAD_UNCHANGED))
    # plt.show()
