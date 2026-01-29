# UC-TwinLiteFormer
轻量化多任务语义分割网络UC-TwinLiteFormer（图4）。模型整体采用“编码器–特征增强模块CAAM–双分支解码器”的结构框架，兼顾实时性与特征表达能力。 该网络的核心创新在于将基于空洞卷积的多尺度特征提取与轻量级Transformer全局建模机制相结合，从理论上弥补传统CNN在非结构化环境中对远距依赖关系建模不足的问题。同时，设计的STF（Skip-Transformer Fusion）模块与LAFAM（Lightweight Adaptive Feature Alignment Module）协同工作，解决跨层语义不一致与多尺度特征对齐问题。
