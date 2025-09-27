import os, threading, queue, time, json, atexit
from pathlib import Path
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import datetime

THREAD_ENABLED = True
LOG_DIR = Path(os.getenv("INTENT_EMOTION_LOG_DIR", "logs/intent_emotion"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
TIME = datetime.datetime.now().strftime("%m%d_%H%M")
REQUIRED_KEYS = {"input_data","emotion_p","intent_emphi","intent_p", "final_p"}

class BatchIntegrator:
    """按需求：每 eval_per_epoch (例如10) 个 epoch 采集一次该 epoch 内 *所有* batch。

    使用方式（训练循环示例伪码）：
        for epoch in range(E):
            for batch_id, batch in enumerate(dataloader):
                BATCH_INTEGRATOR.get_batch_epoch(batch_id, epoch, eval_per_epoch=10, total_batches=len(dataloader))
                ... forward ...
            # 在最后一个 batch 结束（外层可选）调用：
            if BATCH_INTEGRATOR.need_store(epoch):
                BATCH_INTEGRATOR.store()

    也可以不调用 need_store()，直接在最后一个 batch 后无条件调用 store(); 内部会判定 run_or_not。
    """
    def __init__(self):
        # 保存某个目标 epoch 中所有 batch 的 MatrixEmitter
        self.partial: Dict[str, MatrixEmitter] = {}
        self.run_or_not: bool = False          # 当前 epoch 是否在采集
        self.epoch_id: Optional[int] = None
        self.batch_id: Optional[int] = None
        self.eval_per_epoch: int = 10
        self.total_batches_in_epoch: Optional[int] = None

    def update_status(self, run_or_not: bool):
        self.run_or_not = run_or_not

    def get_batch_epoch(self, batch_id: int, epoch_id: int, eval_per_epoch: int, total_batches: int):
        """在每个 batch 调用，决定是否开启或继续采集，并为该 batch 创建 emitter。

        参数:
            batch_id: 本 epoch 内 batch 序号
            epoch_id: 当前 epoch 序号（从0计）
            eval_per_epoch: 多少个 epoch 触发一次采集（例如10）
            total_batches: 本 epoch 总 batch 数
        """
        self.batch_id = batch_id
        new_epoch = (epoch_id != self.epoch_id)
        if new_epoch:
            # 进入新 epoch，重置与上一 epoch 相关的状态
            self.epoch_id = epoch_id
            self.total_batches_in_epoch = total_batches
            self.partial = {}
            # 判断本 epoch 是否为采集目标
            self.run_or_not = ((epoch_id + 1) % eval_per_epoch == 0)
            self.eval_per_epoch = eval_per_epoch

        if not self.run_or_not:
            return  # 非目标 epoch 不采集

        # 为当前 batch 建立唯一 uid：不再使用时间戳，保证一个 epoch 内可读性与可重现（如需绝对唯一性写入文件时再加时间戳）
        current_uid = f"e{self.epoch_id}_b{batch_id}"
        if current_uid not in self.partial:
            self.partial[current_uid] = MatrixEmitter(current_uid)

    def add(self, key: str, value):
        if not self.run_or_not:
            return
        # 当前 batch 的 uid（按照最新 batch_id）
        current_uid = f"e{self.epoch_id}_b{self.batch_id}"
        emitter = self.partial.get(current_uid)
        if emitter is None:
            # 理论上不会发生，防御性处理
            emitter = MatrixEmitter(current_uid)
            self.partial[current_uid] = emitter
        emitter.add(key, value, current_uid)

    def need_store(self, epoch_id: int) -> bool:
        """可选辅助：判断该 epoch 是否应该在末尾调用 store。"""
        return self.run_or_not and epoch_id == self.epoch_id

    def store(self):
        """在目标 epoch 结束（通常最后一个 batch 后）调用。
        写出该 epoch 内所有 batch 的采集结果，并清空缓存。
        文件命名：e{epoch}_b{last_batch}_{timestamp}.json
        """
        if not self.run_or_not:
            return
        if self.batch_id is None:
            return
        timestamp = TIME
        fname = f"e{self.epoch_id}_b{self.batch_id}_{timestamp}.json"
        convert_json_version = {k: v.store_data for k, v in self.partial.items()}
        with open(LOG_DIR / fname, "w") as f:
            json.dump(convert_json_version, f, indent=2, ensure_ascii=False)
        # 清理状态，防止跨 epoch 滚雪球
        self.partial = {}
        self.run_or_not = False
        self.batch_id = None

class MatrixEmitter:
    _instance = None
    def __init__(self, uid):
        self.uid = uid
        self.store_data = {key: [] for key in REQUIRED_KEYS}
        
    def is_full(self):
        for _, value in self.store_data.items():
            if len(value) < 1:
                return False
        return True
            
    def add(self, key: str, value, batch_uid):
        assert batch_uid == self.uid, f"Batch UID mismatch: {batch_uid} vs {self.uid}"
        
        if key == "input_data":
            user, response, p_intent = value["user"], value["response"], value["p_intent"]
            input_data_list = list()
            for i, data in enumerate(user):
                shrinked_value = {
                    'user_input': data["origin_prompt"],
                    'user_emotion': data["matched_emotion"],
                    'response_input': response[i]["origin_prompt"],
                }
                input_data_list.append(shrinked_value)
            self.store_data[key] = input_data_list
            self.store_data["intent_emphi"] = p_intent.clone().cpu().tolist()
        else:
            self.store_data[key]=value

BATCH_INTEGRATOR = BatchIntegrator()
def get_batch_integrator():
    return BATCH_INTEGRATOR