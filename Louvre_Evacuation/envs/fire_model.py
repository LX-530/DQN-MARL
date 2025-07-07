import numpy as np


class ProgressiveFireModel:
    """180步循序渐进火灾扩散模型"""
    
    def __init__(self, fire_sources=None, max_steps=500):
        self.max_steps = max_steps  # 总扩散步数
        self.current_step = 0
        
        if fire_sources is None:
            # 初始火源配置（较小）
            self.initial_fire_sources = [
                {'center': (19, 15), 'size': (2, 2), 'intensity': 0.6},  # 主火源从小开始
                {'center': (22, 18), 'size': (1, 1), 'intensity': 0.4},  # 次火源
                {'center': (16, 12), 'size': (1, 1), 'intensity': 0.4},  # 第三火源
            ]
        else:
            # 转换FireSource对象为字典格式
            self.initial_fire_sources = []
            for fs in fire_sources:
                if hasattr(fs, 'center') and hasattr(fs, 'size'):
                    # FireSource对象
                    self.initial_fire_sources.append({
                        'center': fs.center,
                        'size': fs.size,
                        'intensity': 0.4  # 默认强度
                    })
                elif isinstance(fs, dict):
                    # 字典格式
                    self.initial_fire_sources.append(fs)
        
        # 最终火源配置（大）- 基于初始火源扩展
        self.final_fire_sources = []
        for i, initial in enumerate(self.initial_fire_sources):
            # 扩大初始火源
            expanded = {
                'center': initial['center'],
                'size': (min(8, initial['size'][0] * 4), min(8, initial['size'][1] * 4)),
                'intensity': min(1.0, initial['intensity'] + 0.4)
            }
            self.final_fire_sources.append(expanded)
        
        # 添加额外的新火源
        additional_sources = [
            {'center': (25, 20), 'size': (5, 5), 'intensity': 0.8},   # 新增火源1
            {'center': (13, 18), 'size': (5, 5), 'intensity': 0.8},   # 新增火源2
            {'center': (20, 10), 'size': (4, 4), 'intensity': 0.7},   # 新增火源3
            {'center': (14, 22), 'size': (4, 4), 'intensity': 0.6},   # 新增火源4
            {'center': (26, 14), 'size': (3, 3), 'intensity': 0.5},   # 新增火源5
        ]
        self.final_fire_sources.extend(additional_sources)
        
        # 扩散参数
        self.base_influence_radius = 5.0      # 初始影响半径
        self.max_influence_radius = 20.0      # 最大影响半径
        self.min_danger_level = 0.05          # 最小危险度
        self.health_loss_coefficient = 40.0   # 健康损失系数
        
        # 计算当前火源状态
        self.current_fire_sources = self._interpolate_fire_sources()
    
    def update_step(self):
        """更新时间步"""
        if self.current_step < self.max_steps:
            self.current_step += 1
            self.current_fire_sources = self._interpolate_fire_sources()
    
    def _interpolate_fire_sources(self):
        """根据当前步数插值计算火源状态"""
        if self.current_step >= self.max_steps:
            return self.final_fire_sources.copy()
        
        # 计算进度比例
        progress = self.current_step / self.max_steps
        
        # 分阶段扩散
        if progress < 0.2:  # 前20%：缓慢启动
            expansion_rate = progress * 2  # 0 -> 0.5
        elif progress < 0.5:  # 20%-50%：加速扩散
            expansion_rate = 0.5 + (progress - 0.2) * 1  #  1.0
        elif progress < 0.8:  # 50%-80%：快速扩散
            expansion_rate = 1.0 + (progress - 0.5) * 0.8   # 0.8
        else:  # 最后20%：达到峰值
            expansion_rate = 1.3 + (progress - 0.8) * 0.5   # 0.5
        
        # 限制扩散率
        expansion_rate = min(expansion_rate, 1.0)
        
        # 插值计算当前火源
        current_sources = []
        
        # 基础火源逐步扩大
        for i, (initial, final) in enumerate(zip(self.initial_fire_sources, 
                                                self.final_fire_sources[:len(self.initial_fire_sources)])):
            # 尺寸插值
            size_x = initial['size'][0] + (final['size'][0] - initial['size'][0]) * expansion_rate
            size_y = initial['size'][1] + (final['size'][1] - initial['size'][1]) * expansion_rate
            
            # 强度插值
            intensity = initial['intensity'] + (final['intensity'] - initial['intensity']) * expansion_rate
            
            current_sources.append({
                'center': initial['center'],
                'size': (int(size_x), int(size_y)),
                'intensity': intensity
            })
        
        # 新火源逐步出现
        if len(self.final_fire_sources) > len(self.initial_fire_sources):
            new_sources_count = len(self.final_fire_sources) - len(self.initial_fire_sources)
            
            for i in range(new_sources_count):
                source_index = len(self.initial_fire_sources) + i
                final_source = self.final_fire_sources[source_index]
                
                # 新火源出现时机
                appear_threshold = 0.3 + (i * 0.15)  # 30%, 45%, 60%, 75%...
                
                if progress >= appear_threshold:
                    # 计算新火源的进度
                    new_progress = (progress - appear_threshold) / (1.0 - appear_threshold)
                    new_progress = min(new_progress, 1.0)
                    
                    # 新火源从小开始
                    size_x = 1 + (final_source['size'][0] - 1) * new_progress
                    size_y = 1 + (final_source['size'][1] - 1) * new_progress
                    intensity = 0.2 + (final_source['intensity'] - 0.2) * new_progress
                    
                    current_sources.append({
                        'center': final_source['center'],
                        'size': (int(size_x), int(size_y)),
                        'intensity': intensity
                    })
        
        return current_sources
    
    def get_current_influence_radius(self):
        """获取当前影响半径"""
        progress = min(self.current_step / self.max_steps, 1.0)
        return self.base_influence_radius + (self.max_influence_radius - self.base_influence_radius) * progress
    
    def get_max_danger(self, position):
        """计算位置的最大危险度"""
        max_danger = 0.0
        
        for fire_source in self.current_fire_sources:
            danger = self._calculate_danger_from_source(position, fire_source)
            max_danger = max(max_danger, danger)
        
        return max_danger
    
    def _calculate_danger_from_source(self, position, fire_source):
        """计算单个火源的危险度"""
        center = fire_source['center']
        size = fire_source['size']
        intensity = fire_source['intensity']
        
        # 计算到火源中心的距离
        distance = np.sqrt((position[0] - center[0])**2 + (position[1] - center[1])**2)
        
        # 火源核心区域
        core_radius = max(size[0], size[1]) / 2.0
        if distance <= core_radius:
            return intensity  # 核心区域最大危险
        
        # 当前影响半径
        current_radius = self.get_current_influence_radius()
        
        # 扩展影响区域
        if distance <= current_radius:
            # 分层危险度计算
            if distance <= current_radius * 0.3:
                base_danger = intensity * 1.0 #0.8
            elif distance <= current_radius * 0.5:
                base_danger = intensity * 0.8
            elif distance <= current_radius * 0.7:
                base_danger = intensity * 0.6
            else:
                base_danger = intensity * 0.4
            
            # 距离衰减
            decay_factor = np.exp(-(distance - core_radius) / 6.0)
            danger = base_danger * decay_factor
            
            return max(danger, self.min_danger_level)
        
        return 0.0
    
    def get_fire_info(self):
        """获取当前火源信息"""
        return {
            'step': self.current_step,
            'max_steps': self.max_steps,
            'progress': self.current_step / self.max_steps,
            'fire_count': len(self.current_fire_sources),
            'influence_radius': self.get_current_influence_radius(),
            'sources': self.current_fire_sources
        }


# 兼容性类
class FireSource:
    def __init__(self, center, size, temp_max=600, co_max=1200):
        self.center = center
        self.size = size
        self.temp_max = temp_max
        self.co_max = co_max


class FireSpreadModel:
    """渐进式火灾传播模型"""
    def __init__(self, fire_sources=None, max_steps=180):
        self.progressive_model = ProgressiveFireModel(fire_sources, max_steps)
    
    def get_max_danger(self, position):
        return self.progressive_model.get_max_danger(position)
    
    def update(self):
        self.progressive_model.update_step()
    
    def get_fire_info(self):
        return self.progressive_model.get_fire_info()


# 为了保持向后兼容，提供一个简化的接口
class SimplifiedFireModel:
    """简化火源模型 - 兼容接口"""
    def __init__(self, fire_sources=None):
        self.fire_spread_model = FireSpreadModel(fire_sources)
    
    def get_max_danger(self, position):
        return self.fire_spread_model.get_max_danger(position)
    
    def update(self):
        self.fire_spread_model.update()
    
    @property
    def fire_sources(self):
        return self.fire_spread_model.progressive_model.current_fire_sources
