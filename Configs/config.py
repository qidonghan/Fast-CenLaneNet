__all__={'training_cfg','Dataprocess_cfg','Model_cfg'}

class training_cfg():
    base_lr = 1e-3

class Dataprocess_cfg():
     imgSize = [256, 512]
     gtSize=[256,512]
     lane_width = dict()
     lane_width['seg'] = 2
     lane_width['ins'] = 2

class Model_cfg():
     DAhead_outputchannel=128
