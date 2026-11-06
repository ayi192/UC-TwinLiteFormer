chanel_img = 3

# 车道线任务类别配置
lane_line_classes = 4  # 0: 正常/背景, 1: 车道线, 2: 石头, 3: 沟壑

sc_ch_dict = {
    "nano": {  'p': 1,
            'q': 1,
            'chanels' : [4,8, 16, 32, 64],
    },
    
    "small": {  'p': 2,
            'q': 3,
            'chanels' : [8,16, 32, 64, 128],
    },

    "medium": {  'p': 3,
            'q': 5,
            'chanels' : [16,32, 64, 128, 256],
    },

    "large": {  'p': 5,
            'q': 7,
            'chanels' : [32,64, 128, 256, 512],
    }
}
