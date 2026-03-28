run_NEW.py为最新版

run_SCAE.py为旧版本，其中主要是分类器初始化频率只在epoch=0, 分类损失不回传

文件	定位
run_test.py	最早版本，配合 Painting91Dataset 和旧版 model，需要显式传入 level1
run_test_SSCAE.py	中间版本，切换到 TCFLDataset + model_SSCAE，增加了批量推理/输出到 txt 的能力，level1 全置 0
run_test_NEW.py	当前主版本，用 model_NEW，加载整个模型对象，功能最完整，对应 run_NEW.py 训练产物
run_test_NEW_SJC.py	SJC 消融实验专用，数据来自 paper_data/SJC/，逐张打印预测结果，不打乱顺序

style_predict.py 专门用来进行webstyle的风格识别
