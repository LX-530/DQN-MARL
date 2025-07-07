#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN智能体实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class DQNNetwork(nn.Module):
    """DQN神经网络"""
    
    def __init__(self, state_size, action_size, hidden_size=512):
        super(DQNNetwork, self).__init__()
        
        # 输入状态大小 (11, 11, 6)
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 计算卷积后的特征大小
        conv_output_size = 11 * 11 * 128  # 11x11x128
        
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, action_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 输入形状处理: (batch_size, 11, 11, 6) -> (batch_size, 6, 11, 11)
        if len(x.shape) == 4:
            # 批量输入: (batch_size, 11, 11, 6) -> (batch_size, 6, 11, 11)
            x = x.permute(0, 3, 1, 2)
        elif len(x.shape) == 3:
            # 单个输入: (11, 11, 6) -> (1, 6, 11, 11)
            x = x.permute(2, 0, 1).unsqueeze(0)
        
        # 确保输入是连续的
        x = x.contiguous()
        
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 展平 - 使用reshape而不是view来避免维度不兼容问题
        x = x.reshape(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_size, action_size, device, config):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # 超参数
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.02)
        self.epsilon_decay = config.get('epsilon_decay', 0.9995)
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.batch_size = config.get('batch_size', 32)
        self.target_update_freq = config.get('target_update_freq', 200)
        self.warmup_steps = config.get('warmup_steps', 1000)
        
        # 神经网络
        self.q_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network = DQNNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 经验回放缓冲区
        memory_size = config.get('memory_size', 50000)
        self.memory = deque(maxlen=memory_size)
        
        # 训练步数
        self.steps = 0
        
        # 同步目标网络
        self.update_target_network()
        
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=False):
        """选择动作 - 修复状态处理逻辑"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # 转换状态格式并确保数据类型正确
        if isinstance(state, np.ndarray):
            # 确保状态是float32类型
            state = state.astype(np.float32)
            state_tensor = torch.from_numpy(state).to(self.device)
        else:
            state_tensor = state.to(self.device)
        
        # 确保状态张量是连续的
        if len(state_tensor.shape) == 3:
            state_tensor = state_tensor.unsqueeze(0)
        
        state_tensor = state_tensor.contiguous()
        
        # 获取Q值
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return np.argmax(q_values.cpu().data.numpy())
    
    def learn(self):
        """从经验中学习"""
        if len(self.memory) < self.batch_size or self.steps < self.warmup_steps:
            return
        
        # 采样批次
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量 - 确保数据类型正确
        states = torch.FloatTensor(np.array(states)).to(self.device).contiguous()
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device).contiguous()
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 下一状态的最大Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.steps += 1
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.steps = checkpoint.get('steps', 0) 