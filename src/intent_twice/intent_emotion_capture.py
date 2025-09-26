import os, threading, queue, time, json, atexit
from pathlib import Path
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

THREAD_ENABLED = True
LOG_DIR = Path(os.getenv("INTENT_EMOTION_LOG_DIR", "logs/intent_emotion"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_KEYS = {"input_data","emotion_p","intent_emphi","intent_p"}

class BatchIntegrator:
    def __init__(self):
        self.partial: Dict[MatrixEmitter] = {}
        self.run_or_not: bool = False
        
    def update_status(self, run_or_not: bool):
        self.run_or_not = run_or_not
        
    def get_batch_epoch(self, batch_id: int, epoch_id: int, eval_per_epoch, total_batches: int):
        self.batch_id = batch_id
        self.epoch_id = epoch_id
        if (epoch_id+1) % eval_per_epoch == 0 and (batch_id+1) == total_batches:
            self.run_or_not = True
        
        self.initialize()
        
    def generate_build_in_uid(self):
        return f"e{self.epoch_id}_b{self.batch_id}_{int(time.time()*1000)}"
    
    def initialize(self):
        self.new_uid = self.generate_build_in_uid()
        if self.run_or_not is False:
            return
        new_store = {
            self.new_uid: MatrixEmitter(self.new_uid)
        }
        self.partial.update(new_store)
        self.current_object = new_store[self.new_uid]
    
    def add(self, key: str, value):
        if self.run_or_not is False:
            return
        self.partial[self.new_uid].add(key, value, self.new_uid)
        
    def store(self):
        convert_json_version = {k:v.store_data for k, v in self.partial.items()}
        with open(LOG_DIR / f"{self.new_uid}.json", "w") as f:
            json.dump(convert_json_version, f, indent=2)
        self.run_or_not = False

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